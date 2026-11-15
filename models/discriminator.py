import torch
import torch.nn as nn
import torch.nn.functional as F

class TrajectoryDiscriminator(nn.Module):
    def __init__(self, seq_len=200, num_nodes=22, input_features=2,
                 hidden_size=32, dropout=0.1,pool_size=16):
        super(TrajectoryDiscriminator, self).__init__()
        self.seq_len = seq_len
        self.num_nodes = num_nodes
        self.input_features = input_features
        self.pool_size = min(pool_size, seq_len)
        pooled_size = self.pool_size * num_nodes * input_features
        self.feature_extractor = nn.Sequential(
            nn.Linear(pooled_size, hidden_size * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size * 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),

            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    def forward(self, traj):

        batch_size = traj.size(0)
        x = traj.permute(0, 2, 3, 1).reshape(
            batch_size, self.num_nodes * self.input_features, -1)
        x = F.adaptive_avg_pool1d(x, self.pool_size)
        x = x.reshape(batch_size, -1)
        features = self.feature_extractor(x)
        validity = self.classifier(features)
        return validity
