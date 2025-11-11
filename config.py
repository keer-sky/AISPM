import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_CONFIG = {
    'obs_len': 90,
    'pred_len': 450,
    'seq_len': 540,
    'num_nodes': 22,
    'batch_size': 32,
    'train_dir': "data/train",
    'val_dir': "data/val",
    'test_dir': "data/test"
}

MODEL_CONFIG = {
    'hidden_size': 32,
    'num_heads': 4,
    'num_stgat_blocks': 1,
    'num_transformer_layers': 1,
    'dropout': 0.1,
    'input_features': 2
}

TRAIN_CONFIG = {
    'learning_rate': 0.0001,
    'num_epochs': 100,
    'patience': 10,
    'x_sum_loss_weight': 0.1,
    'save_freq': 1,
    'checkpoint_dir': 'checkpoints',
    'best_model_path': 'best_gan_model.pth',
    'final_model_path': 'final_model.pth'
}