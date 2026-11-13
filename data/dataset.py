import numpy as np
import torch
from torch.utils.data import Dataset
import os
def compute_global_stats(file_paths, num_nodes):
    all_coords = []
    for file_path in file_paths:
        try:
            data = np.loadtxt(file_path, delimiter='\t')
            if len(data) == 0:
                continue
            unique_frames = np.unique(data[:, 0].astype(int))
            for frame in unique_frames:
                frame_data = data[data[:, 0] == frame]
                nodes = frame_data[:, 1].astype(int)
                for node in nodes:
                    if node < num_nodes:
                        coord = frame_data[frame_data[:, 1] == node, 2:4]
                        if coord.size > 0:
                            if not np.all(coord[0] == 0):
                                all_coords.append(coord[0])
        except Exception:
            continue
    if not all_coords:
        return np.zeros(2), np.ones(2)

    all_coords = np.array(all_coords)
    global_mean = np.mean(all_coords, axis=0)
    global_std = np.std(all_coords, axis=0)
    global_std = np.where(global_std == 0, 1, global_std)
    return global_mean, global_std

class TrajectoryDataset(Dataset):
    def __init__(self, file_paths, global_mean, global_std, obs_len=60, pred_len=140, num_nodes=22):
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.seq_len = obs_len + pred_len
        self.global_mean = global_mean
        self.global_std = global_std
        self.data = []
        self.original_data = []

        for file_path in file_paths:
            try:
                data = np.loadtxt(file_path, delimiter='\t')
                if len(data) == 0:
                    continue

                frames = data[:, 0].astype(int)
                unique_frames = np.unique(frames)
                num_frames = len(unique_frames)

                if num_frames < self.seq_len:
                    continue

                traj_data = np.zeros((self.seq_len, self.num_nodes, 2))
                for i, frame in enumerate(unique_frames[:self.seq_len]):
                    frame_data = data[data[:, 0] == frame]
                    nodes = frame_data[:, 1].astype(int)
                    for node in nodes:
                        if node < self.num_nodes:
                            traj_data[i, node, :] = frame_data[frame_data[:, 1] == node, 2:4]

                self.original_data.append(traj_data.copy())

                normalized_data = (traj_data - self.global_mean) / self.global_std
                self.data.append(normalized_data)

            except Exception:
                continue
    def denormalize_trajectory(self, normalized_data):
        if isinstance(normalized_data, torch.Tensor):
            mean = torch.tensor(self.global_mean, dtype=normalized_data.dtype, device=normalized_data.device)
            std = torch.tensor(self.global_std, dtype=normalized_data.dtype, device=normalized_data.device)
            return normalized_data * std + mean
        else:
            return normalized_data * self.global_std + self.global_mean

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        obs_traj = self.data[idx][:self.obs_len]
        pred_traj = self.data[idx][self.obs_len:]
        return torch.FloatTensor(obs_traj), torch.FloatTensor(pred_traj)
