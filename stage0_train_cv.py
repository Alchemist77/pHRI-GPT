#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage-0 통합 러너 (Subject-wise 5-fold CV)
- encoder + head 를 "한 번에" 학습 (CE loss, binary)
- 저장:
  - encoder.pt : LLM ablation에서 재사용
  - head.pt    : No-LLM classification baseline 표(Acc/F1/PR-AUC)용
  - train_idx.npy / val_idx.npy : 재현성 증빙
  - metrics.json : fold별 best metrics

지원:
- TS: transformer / lstm / gru
- IMG: resnet18 / vgg16 / vitb16 (입력 채널 8)
- FUSION: transformer + resnet18 (요구대로)

Loader:
- interaction_timeseries_loader.RobotSkeletonDataset  (seq[K,D], y, ep_id)  (AffectGPT-style sampling 포함)
- interaction_image_loader.InteractionImageDataset    (A[K,8,H,W], B[K,8,H,W], y, ep_id, ...)  (AffectGPT-style sampling 포함)
"""

import os
import json
import argparse
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

# --- 너가 첨부한 loader 그대로 사용 ---
from interaction_timeseries_loader import RobotSkeletonDataset
from interaction_image_loader import InteractionImageDataset
from sklearn.metrics import confusion_matrix


# torchvision 백본 사용
import torchvision


# =========================================================
# 0) 유틸
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def save_json(path: str, obj: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def parse_subject_id(ep_id: str) -> str:
    """
    episode_id 포맷: beh_per_tri
    예: '3_15_8' -> subject='15'
    """
    s = str(ep_id)
    parts = s.split("_")
    if len(parts) >= 2:
        return parts[1]
    return s

def build_subjectwise_folds(episode_ids: List[str], kfold: int, seed: int) -> List[np.ndarray]:
    """
    subject-wise 5-fold:
    - subject 단위로 fold 배정
    - 한 subject의 모든 episode는 동일 fold에만 존재
    return: folds (각 fold는 val indices)
    """
    subj_to_indices: Dict[str, List[int]] = {}
    for i, ep in enumerate(episode_ids):
        sid = parse_subject_id(ep)
        subj_to_indices.setdefault(sid, []).append(i)

    subjects = list(subj_to_indices.keys())
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    subj_folds = np.array_split(subjects, kfold)

    folds: List[np.ndarray] = []
    for f in range(kfold):
        val_subjects = set(subj_folds[f].tolist())
        val_idx = []
        for sid in val_subjects:
            val_idx.extend(subj_to_indices[sid])
        folds.append(np.array(sorted(val_idx), dtype=np.int64))
    return folds

def make_weighted_sampler_binary(y_bin: List[int]) -> WeightedRandomSampler:
    """
    y_bin: 0/1 list
    클래스 불균형 완화 (간단하고 재현성 좋음)
    """
    binc = np.bincount(np.array(y_bin), minlength=2).astype(np.float32)
    binc = np.clip(binc, 1.0, None)
    w = 1.0 / binc
    sample_w = [float(w[int(y)]) for y in y_bin]
    return WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

def safe_metrics_from_logits(logits: torch.Tensor, y_true: torch.Tensor) -> dict:
    """
    logits: (N,2), y_true: (N,)
    metrics: Acc / F1 / PR-AUC
    """
    prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
    pred = torch.argmax(logits, dim=1).detach().cpu().numpy()
    y = y_true.detach().cpu().numpy()

    acc = float((pred == y).mean())

    # sklearn이 있으면 PR-AUC/F1 정확하게
    try:
        from sklearn.metrics import f1_score, average_precision_score
        f1 = float(f1_score(y, pred, average="binary", zero_division=0))
        prauc = float(average_precision_score(y, prob))
    except Exception:
        # fallback(간단): F1 근사(정확도만), PR-AUC는 NaN
        f1 = float(acc)
        prauc = float("nan")

    return {"acc": acc, "f1": f1, "prauc": prauc}

def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# 1) TS Encoders
# =========================================================
class TSTransformerEncoder(nn.Module):
    def __init__(self, feat_dim: int, latent: int = 256, n_layers: int = 3, n_heads: int = 8,
                 ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.in_fc = nn.Linear(feat_dim, latent)
        layer = nn.TransformerEncoderLayer(
            d_model=latent,
            nhead=n_heads,
            dim_feedforward=latent * ff_mult,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,K,D)
        h = self.in_fc(x)
        h = self.enc(h)
        h = h.mean(dim=1)  # temporal average pooling
        return self.norm(h)

class TSRNNEncoder(nn.Module):
    def __init__(self, feat_dim: int, latent: int = 256, rnn_type: str = "lstm",
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.latent = latent
        self.rnn_type = rnn_type.lower()
        RNN = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn = RNN(
            input_size=feat_dim,
            hidden_size=latent,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(latent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,K,D)
        out, _ = self.rnn(x)   # out: (B,K,H)
        h = out.mean(dim=1)    # avg pooling
        return self.norm(h)

class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


# =========================================================
# 2) IMG Encoders (torchvision 기반, 입력 채널 8 대응, K-frame 평균 pooling)
# =========================================================
class KFrameImageEncoder(nn.Module):
    """
    입력: A [B,K,C,H,W] (C=8)
    처리: (B*K,C,H,W) -> backbone -> feat -> reshape(B,K,feat) -> mean(K)
    """
    def __init__(self, backbone_name: str, in_ch: int = 8, latent: int = 256, pretrained: bool = False):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.in_ch = in_ch
        self.latent = latent

        if self.backbone_name == "resnet18":
            m = torchvision.models.resnet18(weights=None if not pretrained else torchvision.models.ResNet18_Weights.DEFAULT)
            # conv1: 3 -> 8
            m.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            feat_dim = m.fc.in_features
            m.fc = nn.Identity()
            self.backbone = m
            self.proj = nn.Linear(feat_dim, latent)

        elif self.backbone_name == "vgg16":
            m = torchvision.models.vgg16(weights=None if not pretrained else torchvision.models.VGG16_Weights.DEFAULT)
            # 첫 conv: 3 -> 8
            first_conv = m.features[0]
            m.features[0] = nn.Conv2d(in_ch, first_conv.out_channels, kernel_size=first_conv.kernel_size,
                                      stride=first_conv.stride, padding=first_conv.padding, bias=False)
            self.features = m.features
            self.avgpool = m.avgpool
            # vgg16 classifier 첫 Linear 입력은 25088 (224 기준). 하지만 image_size 바뀔 수 있으니 안전하게 adaptive로 처리.
            # 여기서는 flatten 이후 Linear로 latent로 투사:
            self.proj = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(4096, latent)
            )

        elif self.backbone_name == "vitb16":
            # torchvision VisionTransformer
            m = torchvision.models.vit_b_16(weights=None if not pretrained else torchvision.models.ViT_B_16_Weights.DEFAULT)
            # patch embedding conv: 3 -> 8
            # torchvision ViT는 conv_proj 라는 Conv2d를 사용
            if hasattr(m, "conv_proj"):
                old = m.conv_proj
                m.conv_proj = nn.Conv2d(in_ch, old.out_channels, kernel_size=old.kernel_size, stride=old.stride, padding=old.padding, bias=False)
            else:
                raise RuntimeError("torchvision vit_b_16 구조가 예상과 다릅니다 (conv_proj 없음).")

            self.vit = m
            # vit의 hidden_dim은 heads 입력 차원과 동일
            hidden_dim = m.heads.head.in_features if hasattr(m.heads, "head") else m.hidden_dim
            # heads는 분류용이니 제거하고 latent로 투사
            m.heads = nn.Identity()
            self.proj = nn.Linear(hidden_dim, latent)

        else:
            raise ValueError(f"Unsupported image backbone: {backbone_name}")

        self.norm = nn.LayerNorm(latent)

    def _forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,C,H,W)
        return: (N, latent)
        """
        if self.backbone_name == "resnet18":
            feat = self.backbone(x)        # (N, feat_dim)
            z = self.proj(feat)

        elif self.backbone_name == "vgg16":
            # vgg16 features -> avgpool -> flatten -> proj
            feat = self.features(x)
            feat = self.avgpool(feat)
            feat = torch.flatten(feat, 1)
            # 입력 해상도가 224가 아니면 feat dim이 달라질 수 있음.
            # 가장 단순하게: 224로 loader가 resize하니(첨부코드) 기본 224 가정이 안전. 
            z = self.proj(feat)

        elif self.backbone_name == "vitb16":
            # torchvision ViT: 내부 encoder 출력에서 CLS 토큰을 가져오기
            # 참고: VisionTransformer._process_input / class_token / encoder / ln
            vit = self.vit
            x2 = vit._process_input(x)  # (N, num_patches, hidden)
            n = x2.shape[0]
            cls = vit.class_token.expand(n, -1, -1)  # (N,1,hidden)
            x2 = torch.cat([cls, x2], dim=1)         # (N, 1+P, hidden)
            x2 = vit.encoder(x2)
            if hasattr(vit, "ln"):
                x2 = vit.ln(x2)
            cls_feat = x2[:, 0]  # (N, hidden)
            z = self.proj(cls_feat)

        else:
            raise RuntimeError("Unexpected backbone")

        return self.norm(z)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        # A: (B,K,C,H,W)
        B, K, C, H, W = A.shape
        x = A.view(B * K, C, H, W)
        z = self._forward_single(x)              # (B*K, latent)
        z = z.view(B, K, self.latent).mean(1)    # (B, latent)
        return z


# =========================================================
# 3) Fusion Head (TS+IMG concat)
# =========================================================
class FusionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256, dropout: float = 0.1, num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


# =========================================================
# 4) Fusion Dataset (ep_id로 TS/IMG 매칭)
# =========================================================
class FusionDataset(torch.utils.data.Dataset):
    """
    TS dataset: __getitem__ -> (seq, y, ep_id)
    IMG dataset: __getitem__ -> (A, B, y, ep_id, ...)
    매칭: ep_id 기준
    """
    def __init__(self, ts_ds: RobotSkeletonDataset, img_ds: InteractionImageDataset, use_view: str = "A"):
        self.ts_ds = ts_ds
        self.img_ds = img_ds
        self.use_view = use_view.upper()

        # img_ds.episodes: (ep_id, r1, r2, f1, f2, T)
        img_map: Dict[str, int] = {}
        for i, ep in enumerate(img_ds.episodes):
            img_map[str(ep[0])] = i

        pairs = []
        for i in range(len(ts_ds)):
            _, _, ep_id = ts_ds[i]
            ep_id = str(ep_id)
            if ep_id in img_map:
                pairs.append((i, img_map[ep_id], ep_id))

        self.pairs = pairs
        print(f"[FusionDataset] matched pairs = {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i_ts, i_img, ep_id = self.pairs[idx]
        seq, y_ts, _ = self.ts_ds[i_ts]
        A, B, y_img, ep2, *_ = self.img_ds[i_img]

        # label은 동일해야 정상. 혹시 다르면 ts label 우선
        y = int(y_ts)

        if self.use_view == "A":
            img = A
        else:
            img = B

        return seq, img, y, ep_id


# =========================================================
# 5) Train/Eval 루프 (공통)
# =========================================================
@dataclass
class TrainConfig:
    root: str
    label_csv: str
    task: str                 # ts/img/fusion
    ts_backbone: str          # transformer/lstm/gru
    img_backbone: str         # resnet18/vgg16/vitb16
    k_frames: int
    pos_label: int
    kfold: int
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    min_delta: float
    latent: int
    out_root: str
    pretrained: bool
    num_workers: int
    image_size: int


def train_one_fold(
    fold_dir: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_parts: Dict[str, nn.Module],
    optimizer: optim.Optimizer,
    cfg: TrainConfig,
) -> dict:
    device = default_device()

    for m in model_parts.values():
        m.to(device)

    best_f1 = -1.0
    bad = 0
    best_metrics = None

    for ep in range(1, cfg.epochs + 1):
        # ---------------- Train ----------------
        for m in model_parts.values():
            m.train()

        train_loss_sum = 0.0
        train_n = 0
        train_logits_all = []
        train_y_all = []

        for batch in train_loader:
            if cfg.task == "ts":
                seq, y, _epid = batch
                seq = seq.to(device)
                yb = (y == cfg.pos_label).long().to(device)

                z = model_parts["encoder"](seq)
                logits = model_parts["head"](z)

            elif cfg.task == "img":
                A, B, y, ep_id, *_ = batch
                A = A.to(device)
                yb = (y == cfg.pos_label).long().to(device)

                z = model_parts["encoder"](A)
                logits = model_parts["head"](z)

            elif cfg.task == "fusion":
                seq, A, y, _epid = batch
                seq = seq.to(device)
                A = A.to(device)
                yb = (y == cfg.pos_label).long().to(device)

                z_ts = model_parts["encoder_ts"](seq)
                z_img = model_parts["encoder_img"](A)
                z = torch.cat([z_ts, z_img], dim=1)
                logits = model_parts["head"](z)

            else:
                raise ValueError("unknown task")

            loss = nn.functional.cross_entropy(logits, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for mm in model_parts.values() for p in mm.parameters() if p.requires_grad],
                1.0
            )
            optimizer.step()

            train_loss_sum += float(loss.item())
            train_n += 1
            train_logits_all.append(logits.detach().cpu())
            train_y_all.append(yb.detach().cpu())

        train_loss = train_loss_sum / max(1, train_n)
        train_logits_all = torch.cat(train_logits_all, dim=0)
        train_y_all = torch.cat(train_y_all, dim=0)
        train_m = safe_metrics_from_logits(train_logits_all, train_y_all)

        # ---------------- Val ----------------
        for m in model_parts.values():
            m.eval()

        val_loss_sum = 0.0
        val_n = 0
        val_logits_all = []
        val_y_all = []

        with torch.no_grad():
            for batch in val_loader:
                if cfg.task == "ts":
                    seq, y, _epid = batch
                    seq = seq.to(device)
                    yb = (y == cfg.pos_label).long().to(device)

                    z = model_parts["encoder"](seq)
                    logits = model_parts["head"](z)

                elif cfg.task == "img":
                    A, B, y, ep_id, *_ = batch
                    A = A.to(device)
                    yb = (y == cfg.pos_label).long().to(device)

                    z = model_parts["encoder"](A)
                    logits = model_parts["head"](z)

                else:  # fusion
                    seq, A, y, _epid = batch
                    seq = seq.to(device)
                    A = A.to(device)
                    yb = (y == cfg.pos_label).long().to(device)

                    z_ts = model_parts["encoder_ts"](seq)
                    z_img = model_parts["encoder_img"](A)
                    z = torch.cat([z_ts, z_img], dim=1)
                    logits = model_parts["head"](z)

                loss = nn.functional.cross_entropy(logits, yb)
                val_loss_sum += float(loss.item())
                val_n += 1

                val_logits_all.append(logits.cpu())
                val_y_all.append(yb.cpu())

        val_loss = val_loss_sum / max(1, val_n)
        val_logits_all = torch.cat(val_logits_all, dim=0)
        val_y_all = torch.cat(val_y_all, dim=0)
        val_m = safe_metrics_from_logits(val_logits_all, val_y_all)

        print(
            f"[Epoch {ep:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_m['acc']:.3f} train_f1={train_m['f1']:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_m['acc']:.3f} val_f1={val_m['f1']:.3f} val_prauc={val_m['prauc']:.3f}"
        )

        # early stop는 val F1 기준 (기존과 동일)
        improved = (val_m["f1"] > best_f1 + cfg.min_delta)
        if improved:
            best_f1 = val_m["f1"]
            bad = 0
            best_metrics = {"train": train_m, "val": val_m, "train_loss": train_loss, "val_loss": val_loss}

            # confusion matrix (val)
            val_pred = torch.argmax(val_logits_all, dim=1).numpy()
            val_true = val_y_all.numpy()
            cm = confusion_matrix(val_true, val_pred, labels=[0, 1])
            np.save(os.path.join(fold_dir, "confusion_val.npy"), cm)
            save_json(os.path.join(fold_dir, "confusion_val.json"), {"labels":[0,1], "cm": cm.tolist()})

            # save weights
            if cfg.task in ("ts", "img"):
                torch.save(model_parts["encoder"].state_dict(), os.path.join(fold_dir, "encoder.pt"))
                torch.save(model_parts["head"].state_dict(), os.path.join(fold_dir, "head.pt"))
            else:
                torch.save(model_parts["encoder_ts"].state_dict(), os.path.join(fold_dir, "encoder_ts.pt"))
                torch.save(model_parts["encoder_img"].state_dict(), os.path.join(fold_dir, "encoder_img.pt"))
                torch.save(model_parts["head"].state_dict(), os.path.join(fold_dir, "head.pt"))

            save_json(os.path.join(fold_dir, "metrics.json"), {"best": best_metrics})

        else:
            bad += 1
            if bad >= cfg.patience:
                break

    if best_metrics is None:
        best_metrics = {"train": {"acc": float("nan"), "f1": float("nan"), "prauc": float("nan")},
                        "val": {"acc": float("nan"), "f1": float("nan"), "prauc": float("nan")},
                        "train_loss": float("nan"), "val_loss": float("nan")}

    return best_metrics["val"]


# =========================================================
# 6) main: task별 dataset/model 구성 + CV 실행
# =========================================================
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="../dataset_v5")
    ap.add_argument("--label_csv", type=str, default="../dataset_v5/generated_phri_captions_rebalanced.csv")

    ap.add_argument("--task", type=str, choices=["ts", "img", "fusion"], required=True)

    ap.add_argument("--ts_backbone", type=str, default="transformer",
                    choices=["transformer", "lstm", "gru"])
    ap.add_argument("--img_backbone", type=str, default="resnet18",
                    choices=["resnet18", "vgg16", "vitb16"])

    ap.add_argument("--k_frames", type=int, default=16)
    ap.add_argument("--pos_label", type=int, default=1)

    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min_delta", type=float, default=1e-4)

    ap.add_argument("--latent", type=int, default=256)

    ap.add_argument("--out_root", type=str, default="stage0_out")
    ap.add_argument("--pretrained", action="store_true", help="torchvision pretrained weights 사용(기본 OFF)")
    ap.add_argument("--num_workers", type=int, default=32)
    ap.add_argument("--image_size", type=int, default=224, help="IMG loader resize size")

    args = ap.parse_args()

    cfg = TrainConfig(
        root=args.root,
        label_csv=args.label_csv,
        task=args.task,
        ts_backbone=args.ts_backbone,
        img_backbone=args.img_backbone,
        k_frames=args.k_frames,
        pos_label=args.pos_label,
        kfold=args.kfold,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        latent=args.latent,
        out_root=args.out_root,
        pretrained=args.pretrained,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    set_seed(cfg.seed)
    device = default_device()
    print(f"[INFO] device={device}")
    print(f"[INFO] task={cfg.task}  k={cfg.k_frames}  seed={cfg.seed}  kfold={cfg.kfold}  pos_label={cfg.pos_label}")
    print(f"[INFO] ts_backbone={cfg.ts_backbone}  img_backbone={cfg.img_backbone}  pretrained={cfg.pretrained}")

    # ---------------- dataset 준비 ----------------
    if cfg.task == "ts":
        ds = RobotSkeletonDataset(cfg.root, cfg.label_csv, k_frames=cfg.k_frames)
        episode_ids = [ep[0] for ep in ds.episodes]  # (ep_id, follower, skel) 

        # input feat dim 확인
        x0, y0, ep0 = ds[0]
        feat_dim = x0.shape[-1]
        print(f"[TS] feat_dim={feat_dim}, seq_shape={tuple(x0.shape)}")

        # 모델명
        model_name = f"ts_{cfg.ts_backbone}"

    elif cfg.task == "img":
        ds = InteractionImageDataset(cfg.root, cfg.label_csv, k_frames=cfg.k_frames,
                                     image_size=(cfg.image_size, cfg.image_size), seed=cfg.seed)
        episode_ids = [ep[0] for ep in ds.episodes]  # (ep_id, r1,r2,f1,f2,T) 

        # input channel 확인 (A: [K,8,H,W])
        A0, B0, y0, ep0, *_ = ds[0]
        in_ch = A0.shape[1]
        print(f"[IMG] in_ch={in_ch}, A_shape={tuple(A0.shape)}")

        model_name = f"img_{cfg.img_backbone}"

    else:  # fusion
        ts_ds = RobotSkeletonDataset(cfg.root, cfg.label_csv, k_frames=cfg.k_frames)
        img_ds = InteractionImageDataset(cfg.root, cfg.label_csv, k_frames=cfg.k_frames,
                                         image_size=(cfg.image_size, cfg.image_size), seed=cfg.seed)
        ds = FusionDataset(ts_ds, img_ds, use_view="A")

        episode_ids = [ds[i][3] for i in range(len(ds))]  # ep_id
        # dims
        x0, A0, y0, ep0 = ds[0]
        feat_dim = x0.shape[-1]
        in_ch = A0.shape[1]
        print(f"[FUSION] feat_dim={feat_dim}, in_ch={in_ch}, seq={tuple(x0.shape)}, A={tuple(A0.shape)}")

        # model_name = f"fusion_transformer_resnet18"
        model_name = f"fusion_{cfg.ts_backbone}_{cfg.img_backbone}"


    # 저장 루트 (모델명 + k + seed 포함)
    out_dir = ensure_dir(os.path.join(cfg.out_root, cfg.task, f"{model_name}_k{cfg.k_frames}_seed{cfg.seed}"))
    print(f"[SAVE] out_dir={out_dir}")

    # subject-wise folds
    folds = build_subjectwise_folds(episode_ids, kfold=cfg.kfold, seed=cfg.seed)

    # ---------------- CV 루프 ----------------
    fold_metrics = []
    for f in range(cfg.kfold):
        fold_dir = ensure_dir(os.path.join(out_dir, f"fold_{f+1}"))

        val_idx = folds[f]
        train_idx = np.hstack([folds[i] for i in range(cfg.kfold) if i != f]).astype(np.int64)

        np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
        np.save(os.path.join(fold_dir, "val_idx.npy"), val_idx)

        # train/val subset
        train_ds = Subset(ds, train_idx)
        val_ds = Subset(ds, val_idx)

        # train sampler용 binary labels
        y_bin = []
        if cfg.task == "ts":
            for i in train_idx:
                _, y, _ = ds[i]
                yb = 1 if int(y) == cfg.pos_label else 0
                y_bin.append(yb)
        elif cfg.task == "img":
            for i in train_idx:
                A, B, y, ep_id, *_ = ds[i]
                yb = 1 if int(y) == cfg.pos_label else 0
                y_bin.append(yb)
        else:
            for i in train_idx:
                _, _, y, _ = ds[i]
                yb = 1 if int(y) == cfg.pos_label else 0
                y_bin.append(yb)

        sampler = make_weighted_sampler_binary(y_bin)

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

        # 모델 구성
        model_parts = {}
        if cfg.task == "ts":
            if cfg.ts_backbone == "transformer":
                encoder = TSTransformerEncoder(feat_dim=feat_dim, latent=cfg.latent)
            elif cfg.ts_backbone in ("lstm", "gru"):
                encoder = TSRNNEncoder(feat_dim=feat_dim, latent=cfg.latent, rnn_type=cfg.ts_backbone)
            else:
                raise ValueError("bad ts_backbone")

            head = LinearHead(in_dim=cfg.latent, num_classes=2)
            model_parts = {"encoder": encoder, "head": head}

            params = list(encoder.parameters()) + list(head.parameters())

        elif cfg.task == "img":
            encoder = KFrameImageEncoder(backbone_name=cfg.img_backbone, in_ch=8, latent=cfg.latent, pretrained=cfg.pretrained)
            head = LinearHead(in_dim=cfg.latent, num_classes=2)
            model_parts = {"encoder": encoder, "head": head}

            params = list(encoder.parameters()) + list(head.parameters())

        # else:  # fusion (transformer + resnet18만)
        #     encoder_ts = TSTransformerEncoder(feat_dim=feat_dim, latent=cfg.latent)
        #     encoder_img = KFrameImageEncoder(backbone_name="resnet18", in_ch=8, latent=cfg.latent, pretrained=cfg.pretrained)
        #     head = FusionHead(in_dim=cfg.latent * 2, hidden=256)

        #     model_parts = {"encoder_ts": encoder_ts, "encoder_img": encoder_img, "head": head}
        #     params = list(encoder_ts.parameters()) + list(encoder_img.parameters()) + list(head.parameters())
        else:  # fusion
            # TS encoder 선택
            if cfg.ts_backbone == "transformer":
                encoder_ts = TSTransformerEncoder(feat_dim=feat_dim, latent=cfg.latent)
            elif cfg.ts_backbone in ("lstm", "gru"):
                encoder_ts = TSRNNEncoder(
                    feat_dim=feat_dim,
                    latent=cfg.latent,
                    rnn_type=cfg.ts_backbone
                )
            else:
                raise ValueError("bad ts_backbone for fusion")

            # IMG encoder 선택
            encoder_img = KFrameImageEncoder(
                backbone_name=cfg.img_backbone,
                in_ch=8,
                latent=cfg.latent,
                pretrained=cfg.pretrained
            )

            head = FusionHead(in_dim=cfg.latent * 2, hidden=256)
            model_parts = {
                "encoder_ts": encoder_ts,
                "encoder_img": encoder_img,
                "head": head,
            }
            params = (
                list(encoder_ts.parameters()) +
                list(encoder_img.parameters()) +
                list(head.parameters())
            )


        optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        print(f"\n==== Fold {f+1}/{cfg.kfold} : train={len(train_idx)} val={len(val_idx)} ====")
        best_m = train_one_fold(
            fold_dir=fold_dir,
            train_loader=train_loader,
            val_loader=val_loader,
            model_parts=model_parts,
            optimizer=optimizer,
            cfg=cfg,
        )
        fold_metrics.append(best_m)

    # CV 요약 저장
    def _mean_std(key: str) -> Tuple[float, float]:
        vals = [m[key] for m in fold_metrics if m.get(key) is not None]
        return float(np.nanmean(vals)), float(np.nanstd(vals))

    acc_m, acc_s = _mean_std("acc")
    f1_m, f1_s = _mean_std("f1")
    pr_m, pr_s = _mean_std("prauc")

    summary = {
        "task": cfg.task,
        "model_name": model_name,
        "k_frames": cfg.k_frames,
        "seed": cfg.seed,
        "kfold": cfg.kfold,
        "pos_label": cfg.pos_label,
        "mean": {"acc": acc_m, "f1": f1_m, "prauc": pr_m},
        "std": {"acc": acc_s, "f1": f1_s, "prauc": pr_s},
        "folds": fold_metrics,
    }
    save_json(os.path.join(out_dir, "cv_summary.json"), summary)

    print("\n==== CV SUMMARY ====")
    print(f"ACC   : {acc_m:.4f} ± {acc_s:.4f}")
    print(f"F1    : {f1_m:.4f} ± {f1_s:.4f}")
    print(f"PR-AUC: {pr_m:.4f} ± {pr_s:.4f}")
    print(f"[SAVED] {os.path.join(out_dir, 'cv_summary.json')}")


if __name__ == "__main__":
    main()

