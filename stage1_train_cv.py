# stage1_train_cv.py
# ---------------------------------------------------------
# Stage-1 LoRA tuning with frozen Stage-0 encoders (TS / IMG / FUSION)
# - Uses Stage-0 subject-wise 5-fold split indices (train_idx.npy / val_idx.npy)
# - Fixed k=16 (or user arg), fixed LoRA rank (default r=8)
# - Saves: best LoRA adapter, tokenizer, projector, logs, config, copies of idx
# ---------------------------------------------------------

import os, re, json, time, random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from unsloth import FastLanguageModel

# use your loaders
from interaction_timeseries_loader import RobotSkeletonDataset
from interaction_image_loader import InteractionImageDataset

# import Stage-0 model definitions so state_dict matches exactly
import stage0_train_cv as s0
import gc

# -----------------------
# Utils
# -----------------------
def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def save_json(path: str, obj: dict):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def sanitize_name(x: str) -> str:
    x = x.strip().replace(" ", "_")
    x = re.sub(r"[^a-zA-Z0-9_\-\.]+", "", x)
    return x

# -----------------------
# Prompt (DO NOT CHANGE)
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

def build_prompt_text(tokenizer):
    return tokenizer.apply_chat_template(
        PROMPT_MESSAGES,
        tokenize=False,
        add_generation_prompt=True
    )

# -----------------------
# Caption mapping
# -----------------------
def load_ep2cap(label_csv: str, caption_col: str = "caption") -> Dict[str, str]:
    df = pd.read_csv(label_csv)
    if "episode_id" not in df.columns:
        raise ValueError("label_csv must contain column 'episode_id'")
    if caption_col not in df.columns:
        raise ValueError(f"label_csv must contain caption column '{caption_col}'")
    ep2cap = {str(r["episode_id"]): str(r[caption_col]) for _, r in df.iterrows()}
    return ep2cap


# -----------------------
# Datasets for Stage-1 (return caption)
# -----------------------
class TSStage1Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, label_csv: str, k_frames: int, ep2cap: Dict[str, str]):
        self.ds = RobotSkeletonDataset(root, label_csv, k_frames=k_frames)
        self.ep2cap = ep2cap

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        seq, y, ep_id = self.ds[i]
        ep_id = str(ep_id)
        cap = self.ep2cap[ep_id]
        return seq, ep_id, cap

def ts_collate(batch):
    seq, ep, cap = zip(*batch)
    return torch.stack(seq, dim=0), list(ep), list(cap)


class IMGStage1Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, label_csv: str, k_frames: int, image_size: int, seed: int,
                 ep2cap: Dict[str, str], use_view: str = "A"):
        self.ds = InteractionImageDataset(
            root=root,
            label_csv=label_csv,
            k_frames=k_frames,
            image_size=(image_size, image_size),
            seed=seed,
        )
        self.ep2cap = ep2cap
        self.use_view = use_view.upper()

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        A, B, y, ep_id, *_ = self.ds[i]
        ep_id = str(ep_id)
        cap = self.ep2cap[ep_id]
        img = A if self.use_view == "A" else B
        return img, ep_id, cap

def img_collate(batch):
    img, ep, cap = zip(*batch)
    return torch.stack(img, dim=0), list(ep), list(cap)


class FUSIONStage1Dataset(torch.utils.data.Dataset):
    def __init__(self, root: str, label_csv: str, k_frames: int, image_size: int, seed: int,
                 ep2cap: Dict[str, str], use_view: str = "A"):
        ts_ds = RobotSkeletonDataset(root, label_csv, k_frames=k_frames)
        img_ds = InteractionImageDataset(root, label_csv, k_frames=k_frames, image_size=(image_size, image_size), seed=seed)
        self.ds = s0.FusionDataset(ts_ds, img_ds, use_view=use_view)
        self.ep2cap = ep2cap

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        seq, img, y, ep_id = self.ds[i]
        ep_id = str(ep_id)
        cap = self.ep2cap[ep_id]
        return seq, img, ep_id, cap

def fusion_collate(batch):
    seq, img, ep, cap = zip(*batch)
    return torch.stack(seq, dim=0), torch.stack(img, dim=0), list(ep), list(cap)


# -----------------------
# Stage-0 encoder builders (must match stage0_train_cv.py)
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
# Projector: z -> prefix embeddings
# -----------------------
class ZProjector(nn.Module):
    def __init__(self, z_dim: int, hidden: int, n_prefix: int):
        super().__init__()
        self.n_prefix = n_prefix
        self.proj = nn.Sequential(
            nn.Linear(z_dim, hidden * n_prefix),
            nn.LayerNorm(hidden * n_prefix),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (B, z_dim) -> (B, n_prefix, hidden)
        B = z.size(0)
        return self.proj(z).view(B, self.n_prefix, -1)

# -----------------------
# Build injected batch embeddings
# -----------------------
def make_injected_batch(
    model,
    tokenizer,
    prompt_text: str,
    prompt_len: int,
    robot_token_id: int,
    n_prefix: int,
    prefix_emb: torch.Tensor,   # (B, n_prefix, hidden) dtype same as model
    captions: List[str],
    max_len: int,
    device: str,
):
    """
    Build inputs_embeds / attention_mask / labels with <|robot|> replaced by prefix_emb.
    """
    texts = [prompt_text + c.strip() + tokenizer.eos_token for c in captions]
    tok = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_len,
    ).to(device)

    input_ids = tok.input_ids
    attn = tok.attention_mask

    # embeddings of original tokens
    emb = model.get_input_embeddings()(input_ids)

    # labels: only caption region contributes, prompt is masked
    labels = input_ids.clone()
    labels[:, :prompt_len] = -100
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    new_emb_list = []
    new_lab_list = []
    new_attn_list = []

    B = input_ids.size(0)
    for b in range(B):
        # find robot token position (must exist exactly once)
        pos = (input_ids[b] == robot_token_id).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            raise RuntimeError("Robot token <|robot|> not found in tokenized prompt. Check chat template.")
        idx = int(pos[0].item())

        # embeddings: replace 1 token with n_prefix prefix embeddings
        new_emb = torch.cat([emb[b, :idx], prefix_emb[b], emb[b, idx+1:]], dim=0)

        # labels: prefix is masked (-100)
        prefix_labels = torch.full((n_prefix,), -100, device=device, dtype=labels.dtype)
        new_lab = torch.cat([labels[b, :idx], prefix_labels, labels[b, idx+1:]], dim=0)

        # attention: prefix is valid tokens (1s)
        prefix_attn = torch.ones((n_prefix,), device=device, dtype=attn.dtype)
        new_attn = torch.cat([attn[b, :idx], prefix_attn, attn[b, idx+1:]], dim=0)

        new_emb_list.append(new_emb)
        new_lab_list.append(new_lab)
        new_attn_list.append(new_attn)

    # stack (length is equal across batch due to padding)
    inputs_embeds = torch.stack(new_emb_list, dim=0)
    labels2 = torch.stack(new_lab_list, dim=0)
    attn2 = torch.stack(new_attn_list, dim=0)
    return inputs_embeds, attn2, labels2


# -----------------------
# Train/Eval for one fold
# -----------------------
@torch.no_grad()
def eval_val_loss(model, tokenizer, prompt_text, prompt_len, robot_id, projector,
                  encoders: Dict[str, nn.Module],
                  val_loader, task: str, n_prefix: int, max_len: int, device: str, amp_dtype):
    model.eval()
    projector.eval()
    for e in encoders.values():
        e.eval()

    total = 0.0
    n = 0
    for batch in val_loader:
        if task == "ts":
            seq, ep, cap = batch
            seq = seq.to(device)
            with torch.no_grad():
                z = encoders["encoder"](seq)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
            z = z.to(model.dtype)
            prefix = projector(z)

        elif task == "img":
            img, ep, cap = batch
            img = img.to(device)
            with torch.no_grad():
                z = encoders["encoder"](img)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
            z = z.to(model.dtype)
            prefix = projector(z)

        else:
            seq, img, ep, cap = batch
            seq = seq.to(device)
            img = img.to(device)
            with torch.no_grad():
                z_ts = encoders["encoder_ts"](seq)
                z_img = encoders["encoder_img"](img)
            z = torch.cat([z_ts, z_img], dim=1)
            z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
            z = z.to(model.dtype)
            prefix = projector(z)

        inputs_embeds, attn, labels = make_injected_batch(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            prompt_len=prompt_len,
            robot_token_id=robot_id,
            n_prefix=n_prefix,
            prefix_emb=prefix,
            captions=cap,
            max_len=max_len,
            device=device,
        )

        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device=="cuda")):
            out = model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)

        total += float(out.loss.item())
        n += 1

    model.train()
    projector.train()
    return total / max(1, n)


def train_one_fold(
    fold_dir: str,
    task: str,
    model,
    tokenizer,
    prompt_text: str,
    prompt_len: int,
    robot_id: int,
    encoders: Dict[str, nn.Module],
    projector: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    min_delta: float,
    max_len: int,
    device: str,
    amp_dtype,
):
    # optimizer: only trainable params (LoRA + projector)
    trainable = []
    for p in model.parameters():
        if p.requires_grad:
            trainable.append(p)
    opt = torch.optim.AdamW(
        [
            {"params": trainable, "lr": lr},
            {"params": projector.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]
    )

    best_val = float("inf")
    bad = 0

    log_path = os.path.join(fold_dir, "train_log.jsonl")

    for ep in range(1, epochs + 1):
        model.train()
        projector.train()
        for e in encoders.values():
            e.eval()  # always frozen encoders

        train_loss_sum = 0.0
        n = 0

        for batch in train_loader:
            if task == "ts":
                seq, ep_ids, caps = batch
                seq = seq.to(device)
                with torch.no_grad():
                    z = encoders["encoder"](seq)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
                z = z.to(model.dtype)
                prefix = projector(z)

            elif task == "img":
                img, ep_ids, caps = batch
                img = img.to(device)
                with torch.no_grad():
                    z = encoders["encoder"](img)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
                z = z.to(model.dtype)
                prefix = projector(z)

            else:
                seq, img, ep_ids, caps = batch
                seq = seq.to(device)
                img = img.to(device)
                with torch.no_grad():
                    z_ts = encoders["encoder_ts"](seq)
                    z_img = encoders["encoder_img"](img)
                z = torch.cat([z_ts, z_img], dim=1)
                z = z / (z.norm(dim=1, keepdim=True) + 1e-6)
                z = z.to(model.dtype)
                prefix = projector(z)

            inputs_embeds, attn, labels = make_injected_batch(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                prompt_len=prompt_len,
                robot_token_id=robot_id,
                n_prefix=projector.n_prefix,
                prefix_emb=prefix,
                captions=caps,
                max_len=max_len,
                device=device,
            )

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(device=="cuda")):
                out = model(inputs_embeds=inputs_embeds, attention_mask=attn, labels=labels)
                loss = out.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            train_loss_sum += float(loss.item())
            n += 1

        train_loss = train_loss_sum / max(1, n)
        val_loss = eval_val_loss(
            model, tokenizer, prompt_text, prompt_len, robot_id,
            projector, encoders, val_loader, task,
            projector.n_prefix, max_len, device, amp_dtype
        )

        improved = (best_val - val_loss) > min_delta
        if improved:
            best_val = val_loss
            bad = 0

            best_dir = ensure_dir(os.path.join(fold_dir, "best"))
            # save lora adapter + tokenizer + projector
            model.save_pretrained(os.path.join(best_dir, "lora_adapter"))
            tokenizer.save_pretrained(os.path.join(best_dir, "tokenizer"))
            torch.save(projector.state_dict(), os.path.join(best_dir, "projector.pt"))
            save_json(os.path.join(best_dir, "best_metrics.json"), {"best_val_loss": best_val})

        else:
            bad += 1

        rec = {"epoch": ep, "train_loss": train_loss, "val_loss": val_loss, "best_val": best_val}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[Epoch {ep:03d}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | best={best_val:.4f} | bad={bad}/{patience}")

        if bad >= patience:
            break

    return best_val


# -----------------------
# Main
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
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--kfold", type=int, default=5)

    ap.add_argument("--latent", type=int, default=256)  # must match stage0 latent
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--stage0_root", type=str, default="./stage0_out")
    ap.add_argument("--out_root", type=str, default="./stage1_out")

    ap.add_argument("--llm_path", type=str, required=True)
    ap.add_argument("--max_len", type=int, default=512)

    ap.add_argument("--n_prefix", type=int, default=8)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.02)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--min_delta", type=float, default=1e-4)
    ap.add_argument("--pretrained_img", action="store_true")  # should match stage0 setting if you used it

    args = ap.parse_args()

    set_seed(args.seed)
    device = default_device()
    print(f"[INFO] device={device}")

    # AMP dtype choice
    # If your GPU supports bf16, this is usually best for Qwen.
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    print(f"[INFO] autocast dtype={amp_dtype}")

    # -------- Stage-0 experiment dir name (must match stage0_train_cv.py convention)
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

    # -------- Output folder
    llm_tag = sanitize_name(os.path.basename(args.llm_path.rstrip("/")))
    out_exp = ensure_dir(os.path.join(
        args.out_root, args.task,
        f"{model_name}_k{args.k_frames}_seed{args.seed}",
        f"llm_{llm_tag}_lora{args.lora_r}_prefix{args.n_prefix}"
    ))
    print(f"[SAVE] out_exp={out_exp}")

    # -------- Load captions
    ep2cap = load_ep2cap(args.label_csv, args.caption_col)

    # -------- Build dataset (full) once; folds use Subset with stage0 indices
    if args.task == "ts":
        ds = TSStage1Dataset(args.root, args.label_csv, args.k_frames, ep2cap)
        # infer feat_dim
        x0, ep0, c0 = ds[0]
        feat_dim = x0.shape[-1]

    elif args.task == "img":
        ds = IMGStage1Dataset(args.root, args.label_csv, args.k_frames, args.image_size, args.seed, ep2cap, use_view="A")
        A0, ep0, c0 = ds[0]
        in_ch = A0.shape[1]

    else:
        ds = FUSIONStage1Dataset(args.root, args.label_csv, args.k_frames, args.image_size, args.seed, ep2cap, use_view="A")
        x0, A0, ep0, c0 = ds[0]
        feat_dim = x0.shape[-1]
        in_ch = A0.shape[1]

    # -------- Loop folds
    fold_best = []
    for f in range(1, args.kfold + 1):

        # -------- Load LLM + tokenizer (Unsloth)
        # NOTE:
        # - model_name can be a local path OR an Unsloth HF model id.
        # - max_seq_length must be >= args.max_len you use in tokenization/truncation.

        # dtype: bf16 if supported else fp16
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name      = args.llm_path,     # can be local path or "unsloth/Qwen3-4B-Instruct-...."
            max_seq_length  = max(args.max_len, 2048),  # 안전하게 크게
            dtype           = dtype,
            load_in_4bit    = False,             # 너는 A6000 Ada라 16bit로 가도 됨
        )

        # pad token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        # add special token for prefix injection
        tokenizer.add_special_tokens({"additional_special_tokens": ["<|robot|>"]})
        robot_id = tokenizer.convert_tokens_to_ids("<|robot|>")
        model.resize_token_embeddings(len(tokenizer))

        # Freeze all first (projector-only baseline or to ensure only LoRA trains)
        for p in model.parameters():
            p.requires_grad = False

        # Apply LoRA if r>0
        if args.lora_r > 0:
            targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]
            model = FastLanguageModel.get_peft_model(
                model,
                r = args.lora_r,
                target_modules = targets,
                lora_alpha = args.lora_alpha,
                lora_dropout = args.lora_dropout,
                bias = "none",
                use_gradient_checkpointing = "unsloth",  # 핵심 가속 옵션
            )
            model.print_trainable_parameters()
        else:
            print("[INFO] LoRA disabled (projector-only baseline).")

        # model.to(device)
        model.train()


        H = model.config.hidden_size

        # prompt token length (for -100 mask)
        prompt_text = build_prompt_text(tokenizer)
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
        prompt_len = int(prompt_ids.size(1))
        print(f"[INFO] prompt_len(tokens)={prompt_len}")


        fold_dir0 = os.path.join(stage0_exp, f"fold_{f}")
        if not os.path.isdir(fold_dir0):
            raise FileNotFoundError(f"Stage-0 fold dir missing: {fold_dir0}")

        train_idx = np.load(os.path.join(fold_dir0, "train_idx.npy"))
        val_idx = np.load(os.path.join(fold_dir0, "val_idx.npy"))

        fold_out = ensure_dir(os.path.join(out_exp, f"fold_{f}"))
        # copy split for proof/repro
        np.save(os.path.join(fold_out, "train_idx.npy"), train_idx)
        np.save(os.path.join(fold_out, "val_idx.npy"), val_idx)

        nw = 8 # argparse에 추가 (default 8 정도 추천)

        dl_kwargs = dict(
            num_workers=nw,
            pin_memory=True,
            persistent_workers=(nw > 0),
        )

        # prefetch_factor는 num_workers>0일 때만
        if nw > 0:
            dl_kwargs["prefetch_factor"] = 2

        if args.task == "ts":
            train_loader = DataLoader(Subset(ds, train_idx),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    collate_fn=ts_collate,
                                    **dl_kwargs)
            val_loader   = DataLoader(Subset(ds, val_idx),
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    collate_fn=ts_collate,
                                    **dl_kwargs)

        elif args.task == "img":
            train_loader = DataLoader(Subset(ds, train_idx),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    collate_fn=img_collate,
                                    **dl_kwargs)
            val_loader   = DataLoader(Subset(ds, val_idx),
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    collate_fn=img_collate,
                                    **dl_kwargs)

        else:
            train_loader = DataLoader(Subset(ds, train_idx),
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    collate_fn=fusion_collate,
                                    **dl_kwargs)
            val_loader   = DataLoader(Subset(ds, val_idx),
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    collate_fn=fusion_collate,
                                    **dl_kwargs)


        # -------- Load Stage-0 encoder weights (must match stage0 naming)
        encoders = {}
        if args.task == "ts":
            enc = build_stage0_ts_encoder(args.ts_backbone, feat_dim=feat_dim, latent=args.latent)
            enc.load_state_dict(torch.load(os.path.join(fold_dir0, "encoder.pt"), map_location="cpu"))
            enc.to(device).eval()
            for p in enc.parameters():
                p.requires_grad = False
            encoders["encoder"] = enc
            z_dim = args.latent

        elif args.task == "img":
            enc = build_stage0_img_encoder(args.img_backbone, in_ch=in_ch, latent=args.latent, pretrained=args.pretrained_img)
            enc.load_state_dict(torch.load(os.path.join(fold_dir0, "encoder.pt"), map_location="cpu"))
            enc.to(device).eval()
            for p in enc.parameters():
                p.requires_grad = False
            encoders["encoder"] = enc
            z_dim = args.latent

        else:
            # enc_ts = build_stage0_ts_encoder("transformer", feat_dim=feat_dim, latent=args.latent)
            # enc_img = build_stage0_img_encoder("resnet18", in_ch=in_ch, latent=args.latent, pretrained=args.pretrained_img)
            enc_ts = build_stage0_ts_encoder(args.ts_backbone, feat_dim=feat_dim, latent=args.latent)
            enc_img = build_stage0_img_encoder(args.img_backbone, in_ch=in_ch, latent=args.latent, pretrained=args.pretrained_img)
            enc_ts.load_state_dict(torch.load(os.path.join(fold_dir0, "encoder_ts.pt"), map_location="cpu"))
            enc_img.load_state_dict(torch.load(os.path.join(fold_dir0, "encoder_img.pt"), map_location="cpu"))
            enc_ts.to(device).eval()
            enc_img.to(device).eval()
            for p in enc_ts.parameters():
                p.requires_grad = False
            for p in enc_img.parameters():
                p.requires_grad = False
            encoders["encoder_ts"] = enc_ts
            encoders["encoder_img"] = enc_img
            z_dim = args.latent * 2  # concat

        # -------- Projector for this fold
        projector = ZProjector(z_dim=z_dim, hidden=H, n_prefix=args.n_prefix).to(device).to(model.dtype)
        projector.train()

        # -------- Save config (one per fold)
        cfg = {
            "task": args.task,
            "ts_backbone": args.ts_backbone,
            "img_backbone": args.img_backbone,
            "k_frames": args.k_frames,
            "latent": args.latent,
            "n_prefix": args.n_prefix,
            "llm_path": args.llm_path,
            "llm_tag": llm_tag,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": args.target_modules,
            "fold": f,
            "stage0_fold_dir": fold_dir0,
            "prompt_messages": PROMPT_MESSAGES,
            "prompt_len": prompt_len,
            "max_len": args.max_len,
            "seed": args.seed,
        }
        save_json(os.path.join(fold_out, "config.json"), cfg)

        print(f"\n==== Stage-1: {args.task} / {model_name} / fold={f} ====\n")
        best = train_one_fold(
            fold_dir=fold_out,
            task=args.task,
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_text,
            prompt_len=prompt_len,
            robot_id=robot_id,
            encoders=encoders,
            projector=projector,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            patience=args.patience,
            min_delta=args.min_delta,
            max_len=args.max_len,
            device=device,
            amp_dtype=amp_dtype,
        )
        fold_best.append(best)
        # ============================
        # 🔥 CLEANUP (매 fold 끝날 때)
        # ============================
        del model
        del projector
        del encoders
        del train_loader, val_loader
        del tokenizer

        torch.cuda.empty_cache()
        gc.collect()

    save_json(os.path.join(out_exp, "cv_summary_stage1.json"), {
        "fold_best_val_loss": fold_best,
        "mean_best_val_loss": float(np.mean(fold_best)),
        "std_best_val_loss": float(np.std(fold_best)),
    })

    print("\n==== DONE Stage-1 ====")
    print(f"[RESULT] mean best val loss = {float(np.mean(fold_best)):.4f} ± {float(np.std(fold_best)):.4f}")
    print(f"[SAVE] {out_exp}")


if __name__ == "__main__":
    main()

