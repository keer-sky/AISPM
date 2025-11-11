import os
import glob
import torch
from torch.utils.data import DataLoader

from config import device, DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from data.dataset import TrajectoryDataset
from models.generator import TrajectoryPredictor
from models.discriminator import TrajectoryDiscriminator
from training.trainer import GANTrainer


def load_data():
    train_files = sorted(glob.glob(os.path.join(DATA_CONFIG['train_dir'], "*.txt")))
    val_files = sorted(glob.glob(os.path.join(DATA_CONFIG['val_dir'], "*.txt")))
    test_files = sorted(glob.glob(os.path.join(DATA_CONFIG['test_dir'], "*.txt")))

    if not train_files or not val_files:
        raise ValueError("Missing training or validation data")

    train_dataset = TrajectoryDataset(train_files, DATA_CONFIG['obs_len'],
                                      DATA_CONFIG['pred_len'], DATA_CONFIG['num_nodes'])
    val_dataset = TrajectoryDataset(val_files, DATA_CONFIG['obs_len'],
                                    DATA_CONFIG['pred_len'], DATA_CONFIG['num_nodes'])

    test_dataset = None
    test_loader = None
    if test_files:
        test_dataset = TrajectoryDataset(test_files, DATA_CONFIG['obs_len'],
                                         DATA_CONFIG['pred_len'], DATA_CONFIG['num_nodes'])
        test_loader = DataLoader(test_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=DATA_CONFIG['batch_size'], shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_models():
    generator = TrajectoryPredictor(
        obs_len=DATA_CONFIG['obs_len'],
        pred_len=DATA_CONFIG['pred_len'],
        num_nodes=DATA_CONFIG['num_nodes'],
        hidden_size=MODEL_CONFIG['hidden_size'],
        num_heads=MODEL_CONFIG['num_heads'],
        num_stgat_blocks=MODEL_CONFIG['num_stgat_blocks'],
        num_transformer_layers=MODEL_CONFIG['num_transformer_layers']
    ).to(device)

    discriminator = TrajectoryDiscriminator(
        seq_len=DATA_CONFIG['seq_len'],
        num_nodes=DATA_CONFIG['num_nodes'],
        hidden_size=MODEL_CONFIG['hidden_size'] // 2
    ).to(device)

    return generator, discriminator


def test_models(generator, discriminator, train_loader):
    test_batch = next(iter(train_loader))
    test_obs, test_true = test_batch
    test_obs = test_obs.to(device)
    test_true = test_true.to(device)

    test_real_traj = torch.cat([test_obs, test_true], dim=1)
    _ = discriminator(test_real_traj)


def train_models(generator, discriminator, train_loader, val_loader):
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=1e-5)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=1e-5)

    trainer = GANTrainer(generator, discriminator, train_loader, val_loader,
                         g_optimizer, d_optimizer, device, TRAIN_CONFIG)

    train_g_losses = []
    train_d_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(TRAIN_CONFIG['num_epochs']):
        avg_g_loss, avg_d_loss, avg_x_sum_loss, g_loss_adv, g_loss_mse, d_loss_real, d_loss_fake = trainer.train_epoch(epoch)
        avg_val_loss, avg_val_x_sum_loss = trainer.validate()

        train_g_losses.append(avg_g_loss)
        train_d_losses.append(avg_d_loss)
        val_losses.append(avg_val_loss)

        print(f'Epoch {epoch + 1}/{TRAIN_CONFIG["num_epochs"]}:')
        print(f'  G Loss: {avg_g_loss:.6f} (Adv: {g_loss_adv:.6f}, MSE: {g_loss_mse:.6f}, X_Sum: {avg_x_sum_loss:.6f})')
        print(f'  D Loss: {avg_d_loss:.6f} (Real: {d_loss_real:.6f}, Fake: {d_loss_fake:.6f})')
        print(f'  Val Loss: {avg_val_loss:.6f} (X_Sum: {avg_val_x_sum_loss:.6f})')

        if (epoch + 1) % TRAIN_CONFIG['save_freq'] == 0 or epoch == 0:
            trainer.save_checkpoint(epoch, avg_g_loss, avg_d_loss, avg_val_loss, avg_val_x_sum_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            trainer.save_best_model(epoch, best_val_loss)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= TRAIN_CONFIG['patience']:
                break

    trainer.save_final_model(epoch, train_g_losses, train_d_losses, val_losses)

    return train_g_losses, train_d_losses, val_losses, best_val_loss

def main():
    print(f'Using device: {device}')

    train_loader, val_loader, _, _, _, _ = load_data()
    generator, discriminator = create_models()
    test_models(generator, discriminator, train_loader)

    train_models(generator, discriminator, train_loader, val_loader)

    checkpoint = torch.load(TRAIN_CONFIG['best_model_path'])
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])


if __name__ == "__main__":
    main()