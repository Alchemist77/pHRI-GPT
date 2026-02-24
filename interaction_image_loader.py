# interaction_image_loader.py
import os, glob, random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

def _is_dir(p: str) -> bool:
    return os.path.isdir(p)

def _sorted_pngs(folder: str):
    files = glob.glob(os.path.join(folder, "*.png"))
    return sorted(files, key=lambda x: int(os.path.basename(x).split(".")[0]))


def proportional_random_indices(T: int, K: int, rng: random.Random):
    """
    AffectGPT-style: split [0,T) into K bins, pick 1 random index per bin.
    If T < K: return [0..T-1] then pad with last index.
    """
    if T <= 0:
        return [0] * K

    if T < K:
        idxs = list(range(T))
        idxs += [T - 1] * (K - T)
        return idxs

    # K bins over [0, T)
    bins = np.linspace(0, T, K + 1).astype(int)
    idxs = []
    for i in range(K):
        a = int(bins[i])
        b = int(bins[i + 1])
        if b <= a:
            b = a + 1
        j = rng.randrange(a, min(b, T))
        idxs.append(j)
    return idxs


class InteractionImageDataset(Dataset):
    """
    Returns:
      rgbA: [K, C, H, W]  (C = 6 + 2 = 8)  (RGB_gaze 3 + RGB_skel 3 + DEP_gaze 1 + DEP_skel 1)
      rgbB: [K, C, H, W]
      y: int
      ep_id: str
      idxsA: LongTensor[K]
      idxsB: LongTensor[K]
      metaA_paths: list[str] length K (filename only)
      metaB_paths: list[str] length K
    """
    def __init__(self, root, label_csv=None, k_frames=16, image_size=(224, 224), seed=42):
        self.root = root
        self.k = int(k_frames)
        self.image_size = tuple(image_size)
        self.episodes = []  # list of (ep_id, r1, r2, f1, f2, T)
        self.labels = {}
        self.rng = random.Random(seed)

        # load labels (optional)
        if label_csv and os.path.exists(label_csv):
            import pandas as pd
            df = pd.read_csv(label_csv)
            # expect columns: episode_id, label
            for _, r in df.iterrows():
                self.labels[str(r["episode_id"])] = int(r["label"])

        self._index_episodes()
        print(f"[Interaction-IMG] Episodes indexed = {len(self.episodes)}")

    def _index_episodes(self):
        if not _is_dir(self.root):
            raise FileNotFoundError(f"root not found: {self.root}")

        for beh in sorted(os.listdir(self.root)):
            p_beh = os.path.join(self.root, beh)
            if not _is_dir(p_beh):
                continue  # <-- skip 0.zip 같은 파일

            for per in sorted(os.listdir(p_beh)):
                p_per = os.path.join(p_beh, per)
                if not _is_dir(p_per):
                    continue

                for tri in sorted(os.listdir(p_per)):
                    p_tri = os.path.join(p_per, tri)
                    if not _is_dir(p_tri):
                        continue

                    ep_id = f"{beh}_{per}_{tri}"

                    rgb1 = os.path.join(p_tri, "image_data_gaze")
                    rgb2 = os.path.join(p_tri, "image_data_skeleton")
                    d1   = os.path.join(p_tri, "image_depth_data_gaze")
                    d2   = os.path.join(p_tri, "image_depth_data_skeleton")

                    if not all(_is_dir(x) for x in [rgb1, rgb2, d1, d2]):
                        continue

                    r1 = _sorted_pngs(rgb1)
                    r2 = _sorted_pngs(rgb2)
                    f1 = _sorted_pngs(d1)
                    f2 = _sorted_pngs(d2)

                    T = min(len(r1), len(r2), len(f1), len(f2))
                    if T < 2:
                        continue

                    self.episodes.append((ep_id, r1, r2, f1, f2, T))

    def __len__(self):
        return len(self.episodes)

    def _load_rgb(self, p):
        img = Image.open(p).convert("RGB").resize(self.image_size)
        x = np.asarray(img, np.float32) / 255.0  # [H,W,3]
        return x

    def _load_depth(self, p):
        img = Image.open(p).convert("L").resize(self.image_size)
        x = np.asarray(img, np.float32)          # [H,W]
        mx = float(x.max())
        x = x / (mx + 1e-6)                      # normalize per-frame
        return x

    def _make_frame_tensor(self, r1p, r2p, d1p, d2p):
        # rgb concat: [H,W,6]
        rgb = np.concatenate([self._load_rgb(r1p), self._load_rgb(r2p)], axis=-1)
        # dep stack: [H,W,2]
        dep = np.stack([self._load_depth(d1p), self._load_depth(d2p)], axis=-1)
        # total: [H,W,8]
        x = np.concatenate([rgb, dep], axis=-1).astype(np.float32)
        return x

    def __getitem__(self, idx):
        ep_id, r1, r2, f1, f2, T = self.episodes[idx]

        # two independent crops from same episode
        idxsA = proportional_random_indices(T, self.k, self.rng)
        idxsB = proportional_random_indices(T, self.k, self.rng)

        A = []
        B = []
        metaA = []
        metaB = []

        for i in idxsA:
            A.append(self._make_frame_tensor(r1[i], r2[i], f1[i], f2[i]))
            metaA.append(os.path.basename(r1[i]))  # gaze filename, as reference

        for i in idxsB:
            B.append(self._make_frame_tensor(r1[i], r2[i], f1[i], f2[i]))
            metaB.append(os.path.basename(r1[i]))

        A = np.stack(A, axis=0)  # [K,H,W,8]
        B = np.stack(B, axis=0)  # [K,H,W,8]

        # to torch: [K,8,H,W]
        A = torch.from_numpy(A).permute(0, 3, 1, 2).contiguous()
        B = torch.from_numpy(B).permute(0, 3, 1, 2).contiguous()

        y = int(self.labels.get(ep_id, 0))

        return A, B, y, ep_id, torch.tensor(idxsA, dtype=torch.long), torch.tensor(idxsB, dtype=torch.long), metaA, metaB


# quick standalone test
def _stat(x: torch.Tensor):
    return dict(mean=float(x.mean()), std=float(x.std()), min=float(x.min()), max=float(x.max()))

def main():
    ROOT = "../dataset_v5"
    LABELS = "../dataset_v5/generated_phri_captions_new.csv"
    K = 16

    ds = InteractionImageDataset(ROOT, LABELS, k_frames=K, seed=42)

    print("\n==== RANDOM 10 EPISODE CHECK ====\n")
    pick = random.sample(range(len(ds)), k=min(10, len(ds)))
    for j, idx in enumerate(pick):
        A, B, y, ep, idxsA, idxsB, metaA, metaB = ds[idx]
        print(f"[{j}] EP={ep} label={y}  Ashape={tuple(A.shape)}  Bshape={tuple(B.shape)}")
        print(f"    idxsA={idxsA.tolist()}")
        print(f"    idxsB={idxsB.tolist()}")
        # compare first frame stats
        print(f"    A[0] stat={_stat(A[0])}")
        print(f"    B[0] stat={_stat(B[0])}")
        eq0 = torch.equal(A[0], B[0])
        print(f"    A[0]==B[0]? {eq0}  (metaA0={metaA[0]} metaB0={metaB[0]})")
        print()

if __name__ == "__main__":
    main()

