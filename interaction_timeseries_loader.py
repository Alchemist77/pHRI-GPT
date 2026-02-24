import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = "../dataset_v5"
LABEL_CSV = "../dataset_v5/generated_phri_captions_rebalanced.csv"

# Robot dims: 39 per arm → 78 robot + 66 skeleton = 144
# follower only: 39 dims
GROUPS = [
    ("pos_xyz",        slice(0,3)),
    ("ori_rpy",        slice(3,6)),
    ("twist_lin",      slice(6,9)),
    ("twist_ang",      slice(9,12)),
    ("wrench_force",   slice(12,15)),
    ("wrench_torque",  slice(15,18)),
    ("joint_pos",      slice(18,25)),
    ("joint_vel",      slice(25,32)),
    ("joint_torque",   slice(32,39)),
]



class RobotSkeletonDataset(Dataset):
    def __init__(self, root, label_csv, k_frames=32):
        self.root = root
        self.k_frames = k_frames

        print(f"[INFO] Reading label CSV: {label_csv}")
        df = pd.read_csv(label_csv)
        self.label_col = "label"
        self.labels = {str(r["episode_id"]): int(r[self.label_col]) for _,r in df.iterrows()}

        self.episodes = []
        for beh in os.listdir(root):
            p_beh=os.path.join(root,beh)
            if not os.path.isdir(p_beh): continue
            for per in os.listdir(p_beh):
                p_per=os.path.join(p_beh,per)
                if not os.path.isdir(p_per): continue
                for tri in os.listdir(p_per):
                    p_tri=os.path.join(p_beh,per,tri)
                    if not os.path.isdir(p_tri): continue

                    ep_id=f"{beh}_{per}_{tri}"
                    if ep_id not in self.labels: continue

                    follower=os.path.join(p_tri,"panda_state_data_follower")
                    skel=os.path.join(p_tri,"pose_keypoints","pose_sequence.csv")
                    if not(os.path.isdir(follower) and os.path.exists(skel)):
                        continue
                    self.episodes.append((ep_id,follower,skel))

        print(f"[INFO] Total episodes found: {len(self.episodes)}")
        self.group_stats_robot, self.skel_mean, self.skel_std = self._compute_stats()
        print("[INFO] Group-wise normalization computed")


    def _load_robot(self, folder):
        fs=sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.endswith(".json")])
        seq=[]
        for jf in fs:
            J=json.load(open(jf))
            seq.append(
                [J["position"]["x"],J["position"]["y"],J["position"]["z"],
                 J["orientation"]["roll"],J["orientation"]["pitch"],J["orientation"]["yaw"]]
                + list(J["twist"]["linear"].values())
                + list(J["twist"]["angular"].values())
                + list(J["wrench"]["force"].values())
                + list(J["wrench"]["torque"].values())
                + J["joint_position"]
                + J["joint_velocity"]
                + J["joint_torque"] )
        return np.array(seq,dtype=np.float32)


    def _load_skel(self,path):
        df=pd.read_csv(path)
        return df.iloc[:,1:].values.astype(np.float32)


    def _compute_stats(self):
        all_R=[]
        all_S=[]
        for ep_id,F,S in self.episodes:
            try:
                f=self._load_robot(F)
                s=self._load_skel(S)
            except: continue
            T=min(len(f),len(s))
            if T<2: continue
            all_R.append(np.concatenate([f[:T]],axis=1))
            all_S.append(s[:T])

        R=np.concatenate(all_R,axis=0)
        S=np.concatenate(all_S,axis=0)

        group_stats={}
        for name, sl in GROUPS:
            m = R[:, sl].mean(0)
            s = np.maximum(R[:, sl].std(0), 1.0)
            group_stats[name] = (m, s)

        sk_m=S.mean(0)
        sk_s=np.maximum(S.std(0),1.0)

        return group_stats, sk_m, sk_s


    def __len__(self):
        return len(self.episodes)


    def _sample_k_frames(self, T):
        """AffectGPT-style proportional random temporal sampling"""
        K = self.k_frames
        if T <= K:
            return np.arange(T)

        bins = np.linspace(0, T, K+1).astype(int)

        idxs = []
        for i in range(K):
            a = bins[i]
            b = max(bins[i+1], a+1)
            idxs.append(np.random.randint(a, b))

        return np.array(idxs, dtype=int)


    def __getitem__(self,i):
        ep_id,F,S=self.episodes[i]
        y=self.labels[ep_id]

        f=self._load_robot(F)
        sk=self._load_skel(S)
        T=min(len(f),len(sk))
        seq=np.concatenate([f[:T],sk[:T]],axis=1)

        # group-wise normalization
        for name, sl in GROUPS:
            m, s = self.group_stats_robot[name]
            seq[:T, sl] = (seq[:T, sl] - m) / s


        # skeleton normalize
        seq[:T,39:105] = (seq[:T,39:105] - self.skel_mean) / self.skel_std


        # clip outliers
        seq = np.clip(seq, -7.0, +7.0)

        # pad / proportional random sample
        if T < self.k_frames:
            seq=np.vstack([seq,np.tile(seq[-1],(self.k_frames-T,1))])
        else:
            idxs = self._sample_k_frames(T)
            seq = seq[idxs]

        return torch.tensor(seq,dtype=torch.float32), y, ep_id

