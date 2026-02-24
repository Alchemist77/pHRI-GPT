#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-2 generation (CV folds) aligned with stage1_train_cv.py

- Loads Stage-0 frozen encoders from stage0_out/.../fold_i/*.pt
- Loads Stage-1 best artifacts from stage1_out/.../fold_i/best/
    - lora_adapter/ (optional if projector-only)
    - tokenizer/
    - projector.pt
    - best_metrics.json
- Uses fold_i/val_idx.npy as test set (same split as Stage-0/Stage-1)
- Prompt MUST match Stage-1 (uses same messages)
- Generation method:
    - tokenize(prompt)
    - replace <|robot|> token embedding with z_prefix tokens
    - model.generate(...) with fixed sampling params (same as your examples)
- Saves:
    stage2_out/<task>/<stage1_exp_name>/<fold_i>/stage2_outputs.csv
    stage2_out/<task>/<stage1_exp_name>/stage2_outputs_all_folds.csv
    stage2_out/<task>/<stage1_exp_name>/config_stage2.json
"""

import os, re, json, random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Your loaders
from interaction_timeseries_loader import RobotSkeletonDataset
from interaction_image_loader import InteractionImageDataset

# Import Stage-0 model defs + FusionDataset to ensure exact weight compatibility
import stage0_train_cv as s0


# -----------------------
# Prompt (KEEP)
# -----------------------
PROMPT_MESSAGES = [
    {
        "role": "system",
        "content": "You are a helpful assistant that describes the human experience during a physical human–robot interaction."
    },
    {
        "role": "user",
        "content": "<|robot|>\nDescribe the human's experience during the interaction in one natural English sentence."
    }
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def sanitize_name(x: str) -> str:
    x = x.strip().replace(" ", "_")
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "", x)


def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_ep_maps(label_csv: str, caption_col: str = "caption") -> Tuple[Dict[str, str], Dict[str, int]]:
    df = pd.read_csv(label_csv)
    if "episode_id" not in df.columns:
        raise ValueError("label_csv must contain 'episode_id'")
    if caption_col not in df.columns:
        # fallbacks commonly used
        if "gt_caption" in df.columns:
            caption_col = "gt_caption"
        elif "caption" in df.columns:
            caption_col = "caption"
        else:
            raise ValueError(f"Cannot find caption column '{caption_col}' (or gt_caption/caption) in label_csv.")

    # label column fallback
    lab_col = "label" if "label" in df.columns else None

    ep2cap = {str(r["episode_id"]): str(r[caption_col]) for _, r in df.iterrows()}
    ep2lab = {str(r["episode_id"]): int(r[lab_col]) for _, r in df.iterrows()} if lab_col else {}
    return ep2cap, ep2lab


# -----------------------
# Stage-1 Projector (must match stage1_train_cv.py)
# -----------------------
class ZProjector(nn.Module):
    def __init__(self, z_dim: int, hidden_dim: int, n_prefix: int):
        super().__init__()
        self.n_prefix = n_prefix
        self.proj = nn.Sequential(
            nn.Linear(z_dim, hidden_dim * n_prefix),
            nn.LayerNorm(hidden_dim * n_prefix),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        return self.proj(z).view(B, self.n_prefix, -1)


# -----------------------
# Build prompt text (KEEP chat style)
# -----------------------
def build_prompt_text(tokenizer) -> str:
    return tokenizer.apply_chat_template(
        PROMPT_MESSAGES,
        tokenize=False,
        add_generation_prompt=True
    )


# -----------------------
# Inject z_prefix into prompt embeddings (KEEP method)
# -----------------------
@torch.no_grad()
def inject_z_prefix(
    model,
    tokenizer,
    prompt_text: str,
    z_prefix: torch.Tensor,   # (1, n_prefix, hidden)
    device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize prompt only, replace <|robot|> token with z_prefix embeddings.
    Returns inputs_embeds, attention_mask.
    """
    ROBOT_ID = tokenizer.convert_tokens_to_ids("<|robot|>")

    tok = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = tok.input_ids

    pos = (input_ids[0] == ROBOT_ID).nonzero(as_tuple=True)[0]
    if pos.numel() == 0:
        raise RuntimeError("Prompt does not contain <|robot|> token. Check prompt/chat_template.")
    idx = int(pos[0].item())

    emb = model.get_input_embeddings()(input_ids)  # (1, L, H)
    emb2 = torch.cat([emb[:, :idx], z_prefix, emb[:, idx + 1:]], dim=1)
    attn = torch.ones((1, emb2.size(1)), device=device, dtype=torch.long)
    return emb2, attn


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt_text: str,
    z_prefix: torch.Tensor,   # (1, n_prefix, hidden)
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float
) -> str:
    """
    Generation method aligned with your stage2 examples:
      model.generate(inputs_embeds=..., attention_mask=..., do_sample=..., temperature=..., top_p=..., repetition_penalty=...)
    We decode ONLY the generated continuation (not the prompt).
    """
    inputs_embeds, attn = inject_z_prefix(model, tokenizer, prompt_text, z_prefix, device)

    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # decode only the newly generated tokens (cleaner than decoding full prompt)
    gen_ids = out[:, inputs_embeds.size(1):] if out.size(1) > inputs_embeds.size(1) else out
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


# -----------------------
# Stage-0 encoder builders (exactly compatible with stage0_train_cv.py)
# -----------------------
def build_stage0_ts_encoder(ts_backbone: str, feat_dim: int, latent: int) -> nn.Module:
    ts_backbone = ts_backbone.lower()
    if ts_backbone == "transformer":
        return s0.TSTransformerEncoder(feat_dim=feat_dim, latent=latent, n_layers=3, n_heads=8, ff_mult=2, dropout=0.1)
    elif ts_backbone in ("lstm", "gru"):
        return s0.TSRNNEncoder(feat_dim=feat_dim, latent=latent, rnn_type=ts_backbone, num_layers=2, dropout=0.1)
    else:
        raise ValueError("bad ts_backbone")


def build_stage0_img_encoder(img_backbone: str, in_ch: int, latent: int, pretrained: bool) -> nn.Module:
    return s0.KFrameImageEncoder(backbone_name=img_backbone, in_ch=in_ch, latent=latent, pretrained=pretrained)


# -----------------------
# MAIN
# -----------------------
def main():
    import argparse
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="../dataset_v5")
    ap.add_argument("--label_csv", type=str, default="../dataset_v5/generated_phri_captions_rebalanced.csv")
    ap.add_argument("--caption_col", type=str, default="caption")

    ap.add_argument("--task", type=str, choices=["ts", "img", "fusion"], required=True)
    ap.add_argument("--ts_backbone", type=str, default="transformer", choices=["transformer", "lstm", "gru"])
    ap.add_argument("--img_backbone", type=str, default="resnet18", choices=["resnet18", "vgg16", "vitb16"])

    ap.add_argument("--k_frames", type=int, default=16)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--use_view", type=str, default="A", choices=["A", "B"])  # keep consistent with Stage-1 choice

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--latent", type=int, default=256)
    ap.add_argument("--pretrained_img", action="store_true")

    ap.add_argument("--stage0_root", type=str, default="./stage0_out")
    ap.add_argument("--stage1_root", type=str, default="./stage1_out")
    ap.add_argument("--out_root", type=str, default="./stage2_out")

    ap.add_argument("--llm_path", type=str, required=True)   # choose one LLM to run
    ap.add_argument("--lora_r", type=int, default=8)         # must match Stage-1 exp
    ap.add_argument("--n_prefix", type=int, default=8)        # must match Stage-1 exp

    # generation params (defaults keep your examples)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--do_sample", action="store_true", default=True)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.05)

    ap.add_argument("--num_workers", type=int, default=2)

    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # Stage-0 exp naming convention (must match stage0_train_cv.py)
    if args.task == "ts":
        model_name = f"ts_{args.ts_backbone}"
    elif args.task == "img":
        model_name = f"img_{args.img_backbone}"
    else:
        # model_name = "fusion_transformer_resnet18"
        model_name = f"fusion_{args.ts_backbone}_{args.img_backbone}"


    stage0_exp = os.path.join(args.stage0_root, args.task, f"{model_name}_k{args.k_frames}_seed{args.seed}")
    if not os.path.isdir(stage0_exp):
        raise FileNotFoundError(f"Stage-0 exp not found: {stage0_exp}")

    # Stage-1 exp naming convention (must match stage1_train_cv.py)
    llm_tag = sanitize_name(os.path.basename(args.llm_path.rstrip("/")))
    stage1_exp = os.path.join(
        args.stage1_root, args.task,
        f"{model_name}_k{args.k_frames}_seed{args.seed}",
        f"llm_{llm_tag}_lora{args.lora_r}_prefix{args.n_prefix}",
    )
    if not os.path.isdir(stage1_exp):
        raise FileNotFoundError(f"Stage-1 exp not found: {stage1_exp}\n"
                                f"Check llm_path basename->llm_tag and lora_r/n_prefix/k/seed match Stage-1.")

    # Output dir
    out_exp = ensure_dir(os.path.join(args.out_root, args.task, f"{model_name}_k{args.k_frames}_seed{args.seed}",
                                     f"llm_{llm_tag}_lora{args.lora_r}_prefix{args.n_prefix}"))
    print(f"[SAVE] out_exp={out_exp}")

    ep2cap, ep2lab = load_ep_maps(args.label_csv, args.caption_col)

    # Build full dataset once; fold uses Subset(val_idx)
    if args.task == "ts":
        full_ds = RobotSkeletonDataset(args.root, args.label_csv, k_frames=args.k_frames)
        # infer feat_dim
        x0, _, ep0 = full_ds[0]
        feat_dim = x0.shape[-1]
        in_ch = None

    elif args.task == "img":
        full_ds = InteractionImageDataset(
            root=args.root,
            label_csv=args.label_csv,
            k_frames=args.k_frames,
            image_size=(args.image_size, args.image_size),
            seed=args.seed,
        )
        A0, B0, _, ep0, *_ = full_ds[0]
        in_ch = (A0 if args.use_view == "A" else B0).shape[1]
        feat_dim = None

    else:
        ts_ds = RobotSkeletonDataset(args.root, args.label_csv, k_frames=args.k_frames)
        img_ds = InteractionImageDataset(
            root=args.root,
            label_csv=args.label_csv,
            k_frames=args.k_frames,
            image_size=(args.image_size, args.image_size),
            seed=args.seed,
        )
        full_ds = s0.FusionDataset(ts_ds, img_ds, use_view=args.use_view)
        x0, A0, _, ep0 = full_ds[0]
        feat_dim = x0.shape[-1]
        in_ch = A0.shape[1]

    # Fold loop
    all_rows = []

    # Save stage2 config
    save_json(os.path.join(out_exp, "config_stage2.json"), {
        "task": args.task,
        "ts_backbone": args.ts_backbone,
        "img_backbone": args.img_backbone,
        "k_frames": args.k_frames,
        "seed": args.seed,
        "kfold": args.kfold,
        "latent": args.latent,
        "n_prefix": args.n_prefix,
        "lora_r": args.lora_r,
        "llm_path": args.llm_path,
        "llm_tag": llm_tag,
        "prompt_messages": PROMPT_MESSAGES,
        "generation": {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
        }
    })

    # =========================
    # Method A: tokenizer + base LLM load ONCE
    # =========================
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=True)
    if "<|robot|>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|robot|>"]})

    torch_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    base_model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.eval()

    # prompt text also ONCE
    prompt_text = build_prompt_text(tokenizer)


    for f in range(1, args.kfold + 1):
        print(f"\n==== Stage-2 fold {f}/{args.kfold} ====\n")

        fold0_dir = os.path.join(stage0_exp, f"fold_{f}")
        if not os.path.isdir(fold0_dir):
            raise FileNotFoundError(f"Missing Stage-0 fold dir: {fold0_dir}")

        fold1_dir = os.path.join(stage1_exp, f"fold_{f}")
        best_dir = os.path.join(fold1_dir, "best")
        if not os.path.isdir(best_dir):
            raise FileNotFoundError(f"Missing Stage-1 best dir: {best_dir}")

        # indices (copied during Stage-1 training)
        val_idx_path = os.path.join(fold1_dir, "val_idx.npy")
        if not os.path.exists(val_idx_path):
            # fallback: from Stage-0
            val_idx_path = os.path.join(fold0_dir, "val_idx.npy")
        val_idx = np.load(val_idx_path).astype(int)

        fold_out = ensure_dir(os.path.join(out_exp, f"fold_{f}"))
        np.save(os.path.join(fold_out, "val_idx.npy"), val_idx)

        lora_dir = os.path.join(best_dir, "lora_adapter")
        if os.path.isdir(lora_dir) and os.path.exists(os.path.join(lora_dir, "adapter_config.json")):
            model = PeftModel.from_pretrained(base_model, lora_dir)
        else:
            model = base_model
        model.eval()


        # Load Stage-0 encoders (frozen)
        encoders = {}
        if args.task == "ts":
            enc = build_stage0_ts_encoder(args.ts_backbone, feat_dim=feat_dim, latent=args.latent)
            enc.load_state_dict(torch.load(os.path.join(fold0_dir, "encoder.pt"), map_location="cpu"))
            enc.to(device).eval()
            for p in enc.parameters():
                p.requires_grad = False
            encoders["encoder"] = enc
            z_dim = args.latent

        elif args.task == "img":
            enc = build_stage0_img_encoder(args.img_backbone, in_ch=in_ch, latent=args.latent, pretrained=args.pretrained_img)
            enc.load_state_dict(torch.load(os.path.join(fold0_dir, "encoder.pt"), map_location="cpu"))
            enc.to(device).eval()
            for p in enc.parameters():
                p.requires_grad = False
            encoders["encoder"] = enc
            z_dim = args.latent

        else:
            enc_ts = build_stage0_ts_encoder(args.ts_backbone, feat_dim=feat_dim, latent=args.latent)
            enc_img = build_stage0_img_encoder( args.img_backbone, in_ch=in_ch, latent=args.latent, pretrained=args.pretrained_img)
            enc_ts.load_state_dict(torch.load(os.path.join(fold0_dir, "encoder_ts.pt"), map_location="cpu"))
            enc_img.load_state_dict(torch.load(os.path.join(fold0_dir, "encoder_img.pt"), map_location="cpu"))
            enc_ts.to(device).eval()
            enc_img.to(device).eval()
            for p in enc_ts.parameters():
                p.requires_grad = False
            for p in enc_img.parameters():
                p.requires_grad = False
            encoders["encoder_ts"] = enc_ts
            encoders["encoder_img"] = enc_img
            z_dim = args.latent * 2

        # Projector (from Stage-1 best)
        H = model.config.hidden_size
        projector = ZProjector(z_dim=z_dim, hidden_dim=H, n_prefix=args.n_prefix).to(device).to(model.dtype)
        proj_path = os.path.join(best_dir, "projector.pt")
        if not os.path.exists(proj_path):
            raise FileNotFoundError(f"Missing projector.pt in {best_dir}")
        projector.load_state_dict(torch.load(proj_path, map_location=device))
        projector.eval()

        # DataLoader over val_idx (batch_size=1 to keep generation identical)
        loader = DataLoader(
            Subset(full_ds, val_idx),
            batch_size=1,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        rows = []
        for i, batch in enumerate(loader):
            if args.task == "ts":
                seq, _, ep_id = batch
                ep_id = str(ep_id[0])
                seq = seq.to(device)
                with torch.no_grad():
                    z = encoders["encoder"](seq)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
                z = z.to(model.dtype)

            elif args.task == "img":
                A, B, _, ep_id, *_ = batch
                ep_id = str(ep_id[0])
                img = A if args.use_view == "A" else B
                img = img.to(device)
                with torch.no_grad():
                    z = encoders["encoder"](img)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
                z = z.to(model.dtype)

            else:
                seq, img, _, ep_id = batch
                ep_id = str(ep_id[0])
                seq = seq.to(device)
                img = img.to(device)
                with torch.no_grad():
                    z_ts = encoders["encoder_ts"](seq)
                    z_img = encoders["encoder_img"](img)
                z = torch.cat([z_ts, z_img], dim=1)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
                z = z.to(model.dtype)

            # REAL z vs ZERO z
            with torch.no_grad():
                ztok_real = projector(z)
                ztok_zero = torch.zeros_like(ztok_real)

            gen_real = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                z_prefix=ztok_real,
                device=device,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )
            gen_zero = generate_one(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                z_prefix=ztok_zero,
                device=device,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
            )

            gt = ep2cap.get(ep_id, "")
            lab = ep2lab.get(ep_id, -1)

            print("------------------------------------------------")
            print(f"[fold{f} {i}] {ep_id}")
            print("LABEL :", lab)
            print("GT    :", gt)
            print("REAL  :", gen_real)
            print("ZERO  :", gen_zero)

            row = {
                "fold": f,
                "episode_id": ep_id,
                "label": lab,
                "gt_caption": gt,
                "gen_real": gen_real,
                "gen_zero": gen_zero,
            }
            rows.append(row)
            all_rows.append(row)

        out_csv = os.path.join(fold_out, "stage2_outputs.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"[SAVED] {out_csv}")
        # (optional) free per-fold objects
        if model is not base_model:
            del model
        del projector, encoders
        torch.cuda.empty_cache()


    # Save merged
    out_all = os.path.join(out_exp, "stage2_outputs_all_folds.csv")
    pd.DataFrame(all_rows).to_csv(out_all, index=False)
    print(f"\n[SAVED] {out_all}")
    print(f"[DONE] Stage-2 outputs saved under: {out_exp}")


if __name__ == "__main__":
    main()

