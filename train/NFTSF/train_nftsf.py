#!/usr/bin/env python3
"""
Training script for Normalizing Flow Time Series Forecasting Model.
Converted from train_model_real_data.ipynb for supercomputer use.

Usage:
    python train_model.py --config config.yaml
    or
    python train_model.py --data_path /path/to/data.npy --output_dir ./results
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless servers
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from architecture import create_nfm
from architecture_encoder import preset_stage1, preset_stage2


def parse_args():
    parser = argparse.ArgumentParser(description='Train NF-TSF Model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to input .npy data file (not required when --data_format sde)')
    parser.add_argument('--data_format', type=str, default='multi_sim',
                        choices=['single_sim', 'multi_sim', 'tnf', 'sde', 'sde_presplit'],
                        help='Data format: single_sim, multi_sim, tnf, sde, or sde_presplit')

    # SDE-specific parameters (used when --data_format sde)
    parser.add_argument('--sde_model_key', type=str, default=None,
                        help='SDE model key, e.g. sw_sle_em or sw_gle_oe_em '
                             '(required when --data_format sde)')
    parser.add_argument('--sde_data_dir', type=str, default='../Data/Trajectories',
                        help='Directory containing SDE .npy files '
                             '(default: ../Data/Trajectories)')
    parser.add_argument('--sde_test_id', type=int, default=1,
                        help='Parameter-set ID used when generating SDE data (default: 1)')
    parser.add_argument('--sde_skip', type=int, default=10,
                        help='Skip value used when generating SDE data (default: 10)')
    
    # Model parameters
    parser.add_argument('--n_past', type=int, default=100,
                        help='Number of past time steps (context)')
    parser.add_argument('--n_future', type=int, default=100,
                        help='Number of future time steps to predict')
    parser.add_argument('--flow_blocks', type=int, default=6,
                        help='Number of flow blocks K')
    parser.add_argument('--hidden_units', type=int, default=64,
                        help='Hidden units in spline conditioner networks')
    parser.add_argument('--hidden_layers', type=str, default='1,2',
                        help='Comma-separated list of hidden layer depths per block (e.g. "1,2")')
    parser.add_argument('--tail_bound', type=float, default=30.0,
                        help='Rational-quadratic spline tail bound for A-RQS flow layers')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--use_scheduler', action='store_true', default=True,
                        help='Use learning rate scheduler')
    parser.add_argument('--scheduler_patience', type=int, default=15,
                        help='Patience for LR scheduler')
    parser.add_argument('--scheduler_factor', type=float, default=0.5,
                        help='Factor to reduce LR')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm')
    parser.add_argument('--normalize', action='store_true', default=True,
                        help='Apply z-score normalization to data')
    
    # Segmentation parameters
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for segment extraction')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Mini-batch size for training (0 = full batch)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for models and plots')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Base name for saved model')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save checkpoint every N epochs (0 to disable)')
    
    # Device parameters
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Resume from checkpoint
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth file to resume training from')

    # Regularization / overfitting prevention
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 weight decay for Adam optimizer (0 to disable)')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of segments held out for validation (0 to disable)')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                        help='Epochs without val improvement before stopping (0 to disable)')

    parser.add_argument('--well_positions', nargs='+', type=float, default=None,
                        metavar='Y',
                        help='Y-axis positions of potential well minima to overlay as '
                             'gray dashed lines on the raw-data and prediction plots. '
                             'E.g. --well_positions -1.0 1.0 for a symmetric double-well.')

    parser.add_argument('--landscape', type=str, default=None,
                        help='Landscape/system name for plot titles '
                             '(default: inferred from data filename)')

    parser.add_argument('--plot_future_steps', type=int, default=None,
                        help='Number of future steps to display on prediction plots. '
                             'Must be <= n_future. Defaults to n_future. '
                             'Controls display only — model predictions are unaffected.')

    parser.add_argument('--model_variant', type=str, default='ar',
                        choices=['ar', 'ar_encoder_full', 'ar_encoder_light'],
                        help='Model architecture variant: '
                             '"ar" (default, original A-RQS flow), '
                             '"ar_encoder_full" (GRU encoder + full flow, K=6), '
                             '"ar_encoder_light" (GRU encoder + lighter flow, K=3).')

    return parser.parse_args()


def setup_device(device_arg):
    """Setup and return the compute device."""
    if device_arg == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_arg)
    
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    torch.set_default_device(device)
    return device


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(data_path, data_format):
    """
    Load and reshape data based on format.
    
    Returns:
        reshaped_data: torch.Tensor of shape (Batch, Time)
    """
    print(f"Loading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)
    print(f"  Raw shape: {data.shape}")
    
    tensor_data = torch.tensor(data, dtype=torch.float32)
    
    if data_format == 'single_sim':
        # Shape: (Samples, Time, 2) where [:,:,1] is Y values
        reshaped_data = tensor_data[:, :, 1]
        
    elif data_format == 'multi_sim':
        # Shape: (Time, 1+NumSims) where column 0 is time
        positions_only = tensor_data[:, 1:]  # Skip time column
        reshaped_data = positions_only.T  # Transpose to (Batch, Time)

    elif data_format == 'sde_presplit':
        # Shape: (N_windows, T) — pre-split output of sde_preprocess.py; already (Batch, Time)
        reshaped_data = tensor_data

    elif data_format == 'tnf':
        # Shape: (T, N, F) - general format
        arr = np.asarray(data)
        if arr.ndim == 1:
            arr = arr[:, None, None]
        elif arr.ndim == 2:
            arr = arr[:, None, :]
        
        # Check if first column is time index
        def looks_like_time(col):
            col = np.asarray(col, dtype=float).reshape(-1)
            if not np.all(np.isfinite(col)):
                return False
            dif = np.diff(col)
            return np.all(dif > 0) and np.median(dif) > 0
        
        feature_slice = slice(None)
        if arr.shape[2] >= 2 and looks_like_time(arr[:, 0, 0]):
            feature_slice = slice(1, None)
            print("  Dropping column 0 (detected as time index)")
        
        data_vals = arr[:, :, feature_slice].astype(np.float32)
        # Use first feature from first track
        series = data_vals[:, 0, 0].reshape(-1)
        # Convert to (1, Time) then we'll segment it
        reshaped_data = torch.tensor(series, dtype=torch.float32).unsqueeze(0)
    
    else:
        raise ValueError(f"Unknown data format: {data_format}")
    
    print(f"  Reshaped to: {reshaped_data.shape} (Batch, Time)")
    return reshaped_data


def extract_segments(tracks, n_past, n_future, stride=1):
    """
    Extracts overlapping fixed-length segments from trajectory data using a
    sliding window, producing the training set for the conditional normalizing
    flow model.

    Given B trajectories each of length L, a sliding window of width
    W = n_past + n_future is advanced by `stride` steps along each trajectory.
    For trajectory i, the windows are:
        [s : s + W]   for s = 0, stride, 2*stride, ..., floor((L - W) / stride) * stride

    Each extracted segment is split during training into:
        context  = segment[:n_past]     (conditioning input, the observed past)
        target   = segment[n_past:]     (prediction target, the future to model)

    The total number of segments produced is:
        N_segments = B * floor((L - W) / stride + 1)

    A stride of 1 maximizes data augmentation by generating all possible
    contiguous windows.  Larger strides reduce overlap and dataset size,
    which may be useful to limit memory or reduce correlation between
    training samples.

    Args:
        tracks: tensor of shape (Batch, Time), B trajectories of length L
        n_past: number of past (context) time steps
        n_future: number of future (target) time steps
        stride: step size between consecutive window starts (default 1)

    Returns:
        segments: tensor of shape (N_segments, n_past + n_future)
    """
    n_extrp = n_past + n_future
    segments = []

    num_tracks, length_track = tracks.shape

    for i in range(num_tracks):
        for start in range(0, length_track - n_extrp + 1, stride):
            segment = tracks[i, start:start + n_extrp]
            segments.append(segment)

    if not segments:
        raise ValueError(f"No segments extracted! Check data length ({length_track}) vs n_extrp ({n_extrp})")

    return torch.stack(segments)


def normalize_data(data):
    """
    Applies z-score (standard score) normalization to the dataset:
        x_norm = (x - mu) / sigma

    where mu = mean(data) and sigma = std(data) are computed over all
    elements of the input tensor (global statistics across all trajectories
    and time steps).

    Z-score normalization centers the data at zero with unit variance,
    which stabilizes and accelerates neural network training by ensuring
    that the loss landscape is well-conditioned and gradients are
    appropriately scaled.

    A small epsilon (1e-8) is added to sigma to prevent division by zero
    in the degenerate case of constant-valued data.

    The returned mu and sigma must be stored and reused at test time to
    apply the same normalization to new data and to de-normalize model
    predictions back to the original physical units:
        x_original = x_norm * sigma + mu

    Args:
        data: tensor of any shape (typically (Batch, Time))

    Returns:
        normalized: z-score normalized tensor (same shape as input)
        mean: scalar global mean mu
        std: scalar global standard deviation sigma (with epsilon)
    """
    mean = data.mean()
    std = data.std() + 1e-8
    normalized = (data - mean) / std
    return normalized, mean, std


def train_model(model, segments, n_past, n_future, epochs, learning_rate,
                use_scheduler, scheduler_patience, scheduler_factor,
                grad_clip, device, output_dir, save_interval, batch_size=0,
                resume_checkpoint=None, weight_decay=1e-5,
                val_fraction=0.0, early_stopping_patience=0,
                pre_val_segments=None):
    """
    Trains the conditional normalizing flow model by maximizing the
    log-likelihood of future trajectories conditioned on observed pasts.

    The model learns a conditional density p(x_future | x_past) by
    transforming a simple base distribution (diagonal Gaussian) through
    a sequence of invertible, learnable maps (normalizing flows).

    The training objective is the negative log-likelihood (NLL):
        L = - (1/N) * sum_{i=1}^{N} log p_theta( x_future^(i) | x_past^(i) )

    where theta are the model parameters (flow weights), and the sum is
    over all N training segments.  Minimizing this is equivalent to
    minimizing the KL divergence between the true conditional distribution
    and the model's learned distribution.

    The log-probability log p_theta(x | context) is computed internally
    by the normalizing flow via the change-of-variables formula:
        log p(x) = log p_base(f^{-1}(x)) + log |det(df^{-1}/dx)|

    where f is the composition of all flow layers and p_base is the
    diagonal Gaussian prior.

    Training details:
        - Optimizer: Adam with configurable learning rate
        - LR scheduler: ReduceLROnPlateau (reduces LR by `factor` after
          `patience` epochs of no improvement)
        - Gradient clipping: max norm clipping to prevent exploding
          gradients, common in deep flow models

    Args:
        model: conditional normalizing flow model
        segments: tensor of shape (N_segments, n_past + n_future)
        n_past: number of context time steps
        n_future: number of target time steps
        epochs: number of training epochs
        learning_rate: initial Adam learning rate
        use_scheduler: whether to use ReduceLROnPlateau
        scheduler_patience: epochs to wait before reducing LR
        scheduler_factor: multiplicative factor for LR reduction
        grad_clip: max gradient norm (0 to disable)
        device: torch device
        output_dir: directory for saving checkpoints
        save_interval: save checkpoint every N epochs (0 to disable)

    Returns:
        epoch_list: list of epoch indices [0, 1, ..., len-1]
        loss_list: list of NLL loss values per epoch
    """
    n_extrp = n_past + n_future

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)

    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience
        )

    start_epoch = 0
    loss_list = []
    val_loss_list = []
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_count = 0

    if resume_checkpoint is not None:
        print(f"\nResuming from checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if use_scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"  Restored scheduler state.")
        elif use_scheduler:
            print(f"  WARNING: Checkpoint has no scheduler state. Scheduler starts fresh.")
        if 'loss_history' in checkpoint:
            loss_list = checkpoint['loss_history']
            print(f"  Restored {len(loss_list)} epochs of loss history.")
        if 'val_loss_history' in checkpoint:
            val_loss_list = checkpoint['val_loss_history']
            print(f"  Restored {len(val_loss_list)} epochs of val loss history.")
        print(f"  Resuming from epoch {start_epoch}, loss was {checkpoint.get('loss', 'N/A')}")

    segments = segments.to(device)

    # Train/validation split.
    # pre_val_segments: pre-split validation segments (used for SDE trajectory-level splits).
    # When provided, val_fraction is ignored and no internal shuffle is performed.
    if pre_val_segments is not None:
        train_segments = segments
        val_segments   = pre_val_segments.to(device)
    else:
        n_total = segments.shape[0]
        if val_fraction > 0.0:
            n_val = max(1, int(n_total * val_fraction))
            perm_all = torch.randperm(n_total, device=device)
            val_idx   = perm_all[:n_val]
            train_idx = perm_all[n_val:]
            train_segments = segments[train_idx]
            val_segments   = segments[val_idx]
        else:
            train_segments = segments
            val_segments   = None

    # Split train segments into context (past) and target (future)
    context    = train_segments[:, :n_past]
    samples    = train_segments[:, n_past:n_extrp]
    n_segments = train_segments.shape[0]
    use_batches = batch_size > 0 and batch_size < n_segments

    if val_segments is not None:
        val_context     = val_segments[:, :n_past]
        val_samples     = val_segments[:, n_past:n_extrp]
        n_val_segments  = val_segments.shape[0]
        use_val_batches = batch_size > 0 and batch_size < n_val_segments

    print(f"\nStarting training:")
    print(f"  Train segments: {n_segments}")
    if val_segments is not None:
        print(f"  Val segments:   {n_val_segments}")
    print(f"  Context shape: {context.shape}")
    print(f"  Samples shape: {samples.shape}")
    print(f"  Batch size: {batch_size if use_batches else 'full'}")
    if use_batches:
        print(f"  Batches per epoch: {(n_segments + batch_size - 1) // batch_size}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    if val_segments is not None:
        print(f"  Val fraction: {val_fraction}")
        print(f"  Early stopping patience: {early_stopping_patience}")
    if resume_checkpoint:
        print(f"  Resuming from epoch: {start_epoch}")
        print(f"  Remaining epochs: {epochs - start_epoch}")

    for epoch in tqdm(range(start_epoch, epochs), desc="Training"):
        if use_batches:
            # Shuffle indices each epoch
            perm = torch.randperm(n_segments, device=device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_segments, batch_size):
                idx = perm[start:start + batch_size]
                batch_samples = samples[idx]
                batch_context = context[idx]

                optimizer.zero_grad()
                loss = -model.log_prob(batch_samples, batch_context).mean()

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n!!! Training crashed at epoch {epoch} (loss={loss.item()})")
                    return list(range(len(loss_list))), loss_list, val_loss_list

                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
        else:
            optimizer.zero_grad()
            # NLL loss: negative mean log-likelihood of targets given contexts
            loss = -model.log_prob(samples, context).mean()

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n!!! Training crashed at epoch {epoch} (loss={loss.item()})")
                break

            loss.backward()

            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()
            avg_loss = loss.item()

        loss_list.append(avg_loss)

        # --- Validation pass ---
        if val_segments is not None:
            model.eval()
            with torch.no_grad():
                if use_val_batches:
                    val_loss_accum, n_vb = 0.0, 0
                    for vs in range(0, n_val_segments, batch_size):
                        ve = min(vs + batch_size, n_val_segments)
                        val_loss_accum += -model.log_prob(
                            val_samples[vs:ve], val_context[vs:ve]).mean().item()
                        n_vb += 1
                    avg_val_loss = val_loss_accum / n_vb
                else:
                    avg_val_loss = -model.log_prob(val_samples, val_context).mean().item()
            model.train()
            val_loss_list.append(avg_val_loss)

            if avg_val_loss < best_val_loss:
                best_val_loss    = avg_val_loss
                best_model_state = {k: v.cpu().clone()
                                    for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            scheduler_signal = avg_val_loss
        else:
            scheduler_signal = avg_loss

        if use_scheduler:
            scheduler.step(scheduler_signal)

        # --- Early stopping ---
        if (early_stopping_patience > 0
                and val_segments is not None
                and no_improve_count >= early_stopping_patience):
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no val improvement for {no_improve_count} epochs)")
            break

        # --- Checkpoint save ---
        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            ckpt_data = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'loss_history': loss_list,
                'val_loss_history': val_loss_list,
            }
            if use_scheduler:
                ckpt_data['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(ckpt_data, checkpoint_path)

    if best_model_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        print(f"Restored best model (val_loss={best_val_loss:.6f})")

    return list(range(len(loss_list))), loss_list, val_loss_list


def save_results(model, loss_list, norm_stats, args, output_dir, val_loss_list=None,
                 landscape_name=None, extra_config=None):
    """Save model, loss curve, and configuration."""

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save model weights
    model_path = os.path.join(output_dir, f'{args.model_name}_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    # Save normalization stats
    if norm_stats is not None:
        norm_path = os.path.join(output_dir, f'norm_stats_{timestamp}.npz')
        np.savez(norm_path,
                 mean=norm_stats['mean'].cpu().numpy(),
                 std=norm_stats['std'].cpu().numpy())
        print(f"Saved normalization stats to: {norm_path}")

    # Save loss history
    loss_path = os.path.join(output_dir, f'loss_history_{timestamp}.npy')
    np.save(loss_path, np.array(loss_list))
    print(f"Saved loss history to: {loss_path}")

    if val_loss_list:
        val_loss_path = os.path.join(output_dir, f'val_loss_history_{timestamp}.npy')
        np.save(val_loss_path, np.array(val_loss_list))
        print(f"Saved val loss history to: {val_loss_path}")

    # Save configuration
    config_path = os.path.join(output_dir, f'config_{timestamp}.json')
    config = vars(args).copy()
    config['timestamp'] = timestamp
    config['final_loss'] = loss_list[-1] if loss_list else None
    config['final_val_loss'] = val_loss_list[-1] if val_loss_list else None
    if extra_config:
        config.update(extra_config)
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")

    # Plot and save linear loss curve
    loss_title = 'Training vs Validation Loss'
    if landscape_name:
        loss_title = f"{landscape_name} — {loss_title}"
    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Train Loss')
    if val_loss_list:
        plt.plot(val_loss_list, label='Val Loss', linestyle='--')
    plt.title(loss_title, fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative Log Likelihood', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, f'training_loss_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot to: {plot_path}")

    # Also plot log loss — guard against non-positive values
    loss_array = np.array(loss_list, dtype=float)
    valid_mask = loss_array > 0
    if valid_mask.any():
        log_title = 'Training Loss (Log Scale)'
        if landscape_name:
            log_title = f"{landscape_name} — {log_title}"
        plt.figure(figsize=(10, 6))
        plt.plot(np.where(valid_mask)[0], np.log(loss_array[valid_mask]),
                 label='Train Loss (log)')
        if val_loss_list:
            val_array = np.array(val_loss_list, dtype=float)
            val_valid = val_array > 0
            if val_valid.any():
                plt.plot(np.where(val_valid)[0], np.log(val_array[val_valid]),
                         label='Val Loss (log)', linestyle='--')
        plt.title(log_title, fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Log(NLL)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        log_plot_path = os.path.join(output_dir, f'training_loss_log_{timestamp}.png')
        plt.savefig(log_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved log-scale loss plot to: {log_plot_path}")
    else:
        print("Warning: no positive loss values found; skipping log-scale plot.")

    return model_path, norm_path if norm_stats else None


def plot_raw_data_sample(data, output_dir, timestamp, n_plots=9, well_positions=None,
                         landscape_name=None):
    """Plot n_plots randomly selected raw trajectories from the loaded dataset.

    Called before training starts so the user can visually inspect what the
    raw (un-normalised) training data looks like before any model is involved.

    Parameters
    ----------
    data           : torch.Tensor of shape (Batch, Time), un-normalised data
    output_dir     : directory where the figure will be saved
    timestamp      : string timestamp used in the output filename
    n_plots        : number of trajectories to show (default 9 → 3×3 grid)
    landscape_name : name of the landscape/system for the plot title and y-axis label
    """
    print("\nPlotting raw data sample...")

    num_available = data.shape[0]
    traj_len = data.shape[1]
    n_plots = min(n_plots, num_available)
    random_indices = np.random.choice(num_available, n_plots, replace=False)

    rows = int(np.ceil(np.sqrt(n_plots)))
    cols = int(np.ceil(n_plots / rows))

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    time_steps = np.arange(traj_len)

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n_plots:
            ax.axis('off')
            continue

        idx = random_indices[i]
        traj = data[idx].cpu().numpy()

        # Well lines at zorder=0 so they sit behind the trajectory.
        if well_positions:
            for j, wp in enumerate(well_positions):
                ax.axhline(y=wp, color='gray', linestyle='--', linewidth=1.2,
                           alpha=0.7, zorder=0,
                           label=f"Well {j + 1}" if i == 0 else '_nolegend_')

        # Y-axis label depends on landscape
        y_label = (r'angle $\varphi$ (rad)'
                   if landscape_name and 'alanine' in landscape_name.lower()
                   else r'Position, $x$')

        ax.plot(time_steps, traj, color='steelblue', linewidth=1.0, alpha=0.85)
        ax.set_title(f"Trajectory {idx}", fontsize=12)
        ax.set_xlabel(r'Step, $N$', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.grid(True, alpha=0.3)

    if well_positions:
        axes[0].legend(loc='upper left', fontsize=9)

    suptitle = "Raw Training Data — 9 Random Trajectories"
    if landscape_name:
        suptitle = f"{landscape_name} — {suptitle}"
    plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, f'raw_data_sample_{timestamp}.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved raw data sample to: {path}")


def visualize_predictions(model, data, norm_stats, n_past, n_future, device, output_dir, timestamp,
                          well_positions=None, landscape_name=None, plot_future_steps=None):
    """Generate and save prediction visualization.

    Parameters
    ----------
    landscape_name   : name of the landscape/system for plot titles and y-axis label
    plot_future_steps: number of future steps to display (defaults to n_future).
                       Only the data passed to each subplot is cropped.
    """

    print("\nGenerating prediction visualizations...")

    # Normalize data if we have norm stats
    if norm_stats is not None:
        norm_data = (data - norm_stats['mean']) / norm_stats['std']
        mu_np = norm_stats['mean'].cpu().numpy()
        std_np = norm_stats['std'].cpu().numpy()
    else:
        norm_data = data
        mu_np = 0
        std_np = 1

    pfs = min(plot_future_steps, n_future) if plot_future_steps is not None else n_future

    # Y-axis label depends on landscape
    y_label = (r'angle $\varphi$ (rad)'
               if landscape_name and 'alanine' in landscape_name.lower()
               else r'Position, $x$')

    # Safety check for indices
    num_available = data.shape[0]
    traj_len = data.shape[1]
    n_extrp = n_past + n_future
    n_plots = min(9, num_available)
    random_indices = np.random.choice(num_available, n_plots, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]
        if i >= n_plots:
            ax.axis('off')
            continue

        idx = random_indices[i]

        # Try up to 3 random start positions to avoid spline instability and
        # to show representative segments rather than always t=0.
        samples_norm = None
        start = 0
        for attempt in range(3):
            start = np.random.randint(0, max(1, traj_len - n_extrp + 1))
            past_norm = norm_data[idx, start:start + n_past].unsqueeze(0).to(device)
            past_repeat = past_norm.repeat(500, 1)
            with torch.no_grad():
                try:
                    samples_norm = model.sample(500, past_repeat)[0].cpu().numpy()
                    break  # success
                except AssertionError as e:
                    print(f"  WARNING: Sampling failed for trajectory {idx} start={start} "
                          f"(attempt {attempt+1}/3, spline instability): {e}")

        if samples_norm is None:
            ax.set_title(f"Trajectory {idx} [FAILED]")
            continue

        # De-normalize samples
        samples_real = (samples_norm * std_np) + mu_np

        # Get ground truth for this segment
        real_traj = data[idx, start:start + n_extrp].cpu().numpy()

        # Plotting (relative time axis, cropped to display window)
        future_steps = np.arange(n_past, n_past + pfs)
        all_steps = np.arange(0, n_past + pfs)

        # Slice samples for display only (underlying arrays not changed)
        median_disp = np.median(samples_real[:, :pfs], axis=0)
        lower_disp = np.percentile(samples_real[:, :pfs], 2.5, axis=0)
        upper_disp = np.percentile(samples_real[:, :pfs], 97.5, axis=0)
        lower_50_disp = np.percentile(samples_real[:, :pfs], 25, axis=0)
        upper_50_disp = np.percentile(samples_real[:, :pfs], 75, axis=0)

        # Well lines at zorder=0 so they sit behind all trajectory data.
        if well_positions:
            for j, wp in enumerate(well_positions):
                ax.axhline(y=wp, color='gray', linestyle='--', linewidth=1.2,
                           alpha=0.7, zorder=0,
                           label=f"Well {j + 1}" if i == 0 else '_nolegend_')

        # Plot truth
        ax.plot(all_steps, real_traj[:n_past + pfs], 'k-', linewidth=1.5, label='Truth')

        # Plot prediction
        ax.plot(future_steps, median_disp, color='tab:orange', linewidth=2, label='Pred Median')
        ax.fill_between(future_steps, lower_disp, upper_disp,
                        color='tab:orange', alpha=0.3, label='95% Band')
        ax.fill_between(future_steps, lower_50_disp, upper_50_disp,
                        color='tab:blue', alpha=0.3, label='50% Band')

        ax.set_title(f"Traj {idx} (t={start})", fontsize=12)
        ax.set_xlabel(r'Step, $N$', fontsize=10)
        ax.set_ylabel(y_label, fontsize=10)
        ax.axvline(x=n_past, color='k', linestyle='--', alpha=0.3)

    axes[0].legend(loc='upper left')

    suptitle = "Sample Predictions (Training Data)"
    if landscape_name:
        suptitle = f"{landscape_name} — {suptitle}"
    plt.suptitle(suptitle, fontsize=14, y=1.01)
    plt.tight_layout()

    viz_path = os.path.join(output_dir, f'predictions_{timestamp}.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction visualization to: {viz_path}")


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # SDE data path: validate args early so errors surface before setup.  #
    # ------------------------------------------------------------------ #
    if args.data_format == 'sde':
        if args.sde_model_key is None:
            raise ValueError(
                "--sde_model_key is required when --data_format sde.  "
                "Example: --sde_model_key sw_sle_em"
            )
    elif args.data_path is None:
        raise ValueError(
            "--data_path is required when --data_format is not 'sde'."
        )

    # Determine landscape name for plot titles
    landscape_name = args.landscape
    if landscape_name is None:
        if args.data_format == 'sde':
            landscape_name = args.sde_model_key
        else:
            landscape_name = os.path.splitext(os.path.basename(args.data_path))[0]

    # Determine how many future steps to display on plots (separate from n_future).
    plot_future_steps = args.plot_future_steps if args.plot_future_steps is not None else args.n_future

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup
    set_seed(args.seed)
    device = setup_device(args.device)

    # ------------------------------------------------------------------ #
    # Load data — standard path (single .npy) or SDE path (x + v files). #
    # ------------------------------------------------------------------ #
    pre_val_segments = None   # set only for SDE trajectory-level splits

    if args.data_format == 'sde':
        from sde_loader import load_sde_dataset, split_sde_dataset
        mem_label = '(memory)' if 'gle' in args.sde_model_key else '(no memory)'
        print(f"\nLoading SDE dataset: {args.sde_model_key} {mem_label}")

        sde_dataset = load_sde_dataset(
            model_key=args.sde_model_key,
            data_dir=args.sde_data_dir,
            test_id=args.sde_test_id,
            skip=args.sde_skip,
        )
        sde_split = split_sde_dataset(sde_dataset, random_seed=args.seed)

        x_train_np = sde_split['x_train']   # (N_train, T)  numpy
        x_val_np   = sde_split['x_val']     # (N_val,   T)  numpy
        print(f"  Train trajectories: {x_train_np.shape[0]}  "
              f"Val trajectories: {x_val_np.shape[0]}  "
              f"(test trajectories: {sde_split['x_test'].shape[0]} — held out)")

        # Normalise on train statistics only, then apply to val.
        norm_stats = None
        x_train_t = torch.tensor(x_train_np, dtype=torch.float32)
        x_val_t   = torch.tensor(x_val_np,   dtype=torch.float32)
        if args.normalize:
            norm_mean = x_train_t.mean()
            norm_std  = x_train_t.std() + 1e-8
            norm_stats = {'mean': norm_mean, 'std': norm_std}
            x_train_t = (x_train_t - norm_mean) / norm_std
            x_val_t   = (x_val_t   - norm_mean) / norm_std
            print(f"  Normalized: mean={norm_mean:.4f}  std={norm_std:.4f}")

        # Extract sliding-window segments from train and val separately
        # so the trajectory-level boundary is respected.
        segments         = extract_segments(x_train_t, args.n_past, args.n_future, args.stride)
        pre_val_segments = extract_segments(x_val_t,   args.n_past, args.n_future, args.stride)
        print(f"  Train segments: {segments.shape[0]}  Val segments: {pre_val_segments.shape[0]}")

        # reshaped_data used only for visualisation; use train trajectories.
        reshaped_data = x_train_t
    else:
        # Standard single-.npy load path (unchanged for existing data types).
        reshaped_data = load_data(args.data_path, args.data_format)
        norm_stats = None
        if args.normalize:
            norm_data, mean, std = normalize_data(reshaped_data)
            norm_stats = {'mean': mean, 'std': std}
            print(f"Normalized data: mean={mean:.4f}, std={std:.4f}")
        else:
            norm_data = reshaped_data
        segments = extract_segments(norm_data, args.n_past, args.n_future, args.stride)
        print(f"Extracted {segments.shape[0]} segments of length {segments.shape[1]}")

    # Visualize 9 random raw trajectories before training.
    try:
        timestamp_data = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_raw_data_sample(reshaped_data, args.output_dir, timestamp_data,
                             well_positions=args.well_positions,
                             landscape_name=landscape_name)
    except Exception as e:
        print(f"WARNING: Raw data visualization failed: {e}")

    # Create model
    context_size = args.n_past
    latent_size = args.n_future
    hidden_layers_list = tuple(int(x) for x in args.hidden_layers.split(','))
    print(f"\nModel variant: {args.model_variant}")
    if args.model_variant == "ar":
        model = create_nfm(device, latent_size, context_size,
                            K=args.flow_blocks, hidden_units=args.hidden_units,
                            hidden_layers_list=hidden_layers_list,
                            tail_bound=args.tail_bound)
    elif args.model_variant == "ar_encoder_full":
        model = preset_stage1(device, n_past=args.n_past, n_future=args.n_future, past_dim=1)
    elif args.model_variant == "ar_encoder_light":
        model = preset_stage2(device, n_past=args.n_past, n_future=args.n_future, past_dim=1)
    else:
        raise ValueError(f"Unknown model_variant: {args.model_variant}")

    # Train
    # For SDE: pre_val_segments carries the trajectory-level validation split.
    # For other formats: pre_val_segments=None and val_fraction does internal split.
    _train_start = time.time()
    epoch_list, loss_list, val_loss_list = train_model(
        model=model,
        segments=segments,
        n_past=args.n_past,
        n_future=args.n_future,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        use_scheduler=args.use_scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        grad_clip=args.grad_clip,
        device=device,
        output_dir=args.output_dir,
        save_interval=args.save_interval,
        batch_size=args.batch_size,
        resume_checkpoint=args.resume_checkpoint,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction if pre_val_segments is None else 0.0,
        early_stopping_patience=args.early_stopping_patience,
        pre_val_segments=pre_val_segments,
    )
    _training_time_seconds = time.time() - _train_start
    print(f"\nTotal training time: {_training_time_seconds:.1f} seconds "
          f"({_training_time_seconds/60:.1f} min)")

    # Quick sampling benchmark: time 100 samples from the first segment's context.
    _samp_n = 100
    _samp_ctx = segments[:1, :args.n_past].to(device)      # (1, n_past)
    _samp_rep = _samp_ctx.expand(_samp_n, -1)              # (n_samples, n_past)
    if device.type == "cuda":
        torch.cuda.synchronize()
    _t0 = time.time()
    with torch.no_grad():
        model.sample(_samp_n, _samp_rep)
    if device.type == "cuda":
        torch.cuda.synchronize()
    _sampling_time = time.time() - _t0
    _time_per_sample = _sampling_time / _samp_n
    print(f"Sampling benchmark ({_samp_n} samples): "
          f"{_sampling_time*1000:.1f}ms total  "
          f"({_time_per_sample*1000:.3f}ms/sample)")

    _timing_extra = {
        "training_time_seconds":          _training_time_seconds,
        "sampling_benchmark_n_samples":   _samp_n,
        "sampling_benchmark_total_seconds": _sampling_time,
        "sampling_time_per_sample_seconds": _time_per_sample,
    }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path, norm_path = save_results(model, loss_list, norm_stats, args, args.output_dir,
                                         landscape_name=landscape_name,
                                         extra_config=_timing_extra)

    # Generate visualizations
    try:
        visualize_predictions(model, reshaped_data, norm_stats, args.n_past, args.n_future,
                             device, args.output_dir, timestamp,
                             well_positions=args.well_positions,
                             landscape_name=landscape_name,
                             plot_future_steps=plot_future_steps)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"WARNING: Visualization failed: {e}")
        print("Model and weights were saved successfully.")

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Final train loss: {loss_list[-1]:.6f}")
    if val_loss_list:
        print(f"Final val loss:   {val_loss_list[-1]:.6f}")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()