#!/usr/bin/env python3
"""
Training script for Normalizing Flow Time Series Forecasting Model.
Simplified version: loads .npz with 'positions' key (like CSDI) or .npy.
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "NF"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))
sys.path.insert(0, PROJECT_ROOT)

from architecture import create_nfm
from architecture_encoder import preset_stage1, preset_stage2


def parse_args():
    parser = argparse.ArgumentParser(description='Train NF-TSF Model')

    # Data parameters – simplified: just a path
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to .npz (with "positions" key) or .npy file')

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
                        help='Comma-separated list of hidden layer depths per block')
    parser.add_argument('--tail_bound', type=float, default=30.0,
                        help='Rational-quadratic spline tail bound')

    # Training parameters
    parser.add_argument("--encoder_type", type=str, default="gru")
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
    parser.add_argument('--normalization', type=str, default='none',
                    choices=['none', 'zscore', 'mean_abs'],
                    help='Normalization method: none, zscore (global), mean_abs (per‑series mean abs, like TSDiff)'),
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for segment extraction')
    parser.add_argument('--batch_size', type=int, default=4096,
                        help='Mini-batch size for training (0 = full batch)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='L2 weight decay for Adam')
    parser.add_argument('--val_fraction', type=float, default=0.1,
                        help='Fraction of segments held out for validation (0 to disable)')
    parser.add_argument('--early_stopping_patience', type=int, default=50,
                        help='Epochs without val improvement before stopping (0 to disable)')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory for models and plots')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Base name for saved model')
    parser.add_argument('--save_interval', type=int, default=100,
                        help='Save checkpoint every N epochs (0 to disable)')

    # Device and reproducibility
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help='Path to checkpoint .pth file to resume from')

    # Visualisation options
    parser.add_argument('--well_positions', nargs='+', type=float, default=None,
                        metavar='Y', help='Y positions of well minima for dashed lines')
    parser.add_argument('--landscape', type=str, default=None,
                        help='Landscape name for plot titles')
    parser.add_argument('--plot_future_steps', type=int, default=None,
                        help='Number of future steps to display (defaults to n_future)')

    # Model variant
    parser.add_argument('--model_variant', type=str, default='ar',
                        choices=['ar', 'ar_encoder_full', 'ar_encoder_light'],
                        help='Model architecture variant')

    return parser.parse_args()


def setup_device(device_arg):
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
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_data(data_path):
    """
    Load data from .npz (extracts 'positions' key) or .npy.
    Returns tensor of shape (Batch, Time).
    """
    print(f"Loading data from: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    if data_path.endswith('.npz'):
        if isinstance(data, np.lib.npyio.NpzFile):
            if 'positions' in data:
                arr = data['positions']
            else:
                # fallback: take the first array
                arr = list(data.values())[0]
        else:
            arr = data
    else:
        # Plain .npy
        arr = data

    print(f"  Array shape: {arr.shape}")
    reshaped_data = torch.tensor(arr, dtype=torch.float32)
    print(f"  Loaded as: {reshaped_data.shape} (Batch, Time)")
    return reshaped_data


def extract_segments(tracks, n_past, n_future, stride=1):
    """
    Extract overlapping windows from trajectories.
    Returns tensor of shape (N_segments, n_past + n_future).
    """
    n_extrp = n_past + n_future
    segments = []
    num_tracks, length_track = tracks.shape
    for i in range(num_tracks):
        for start in range(0, length_track - n_extrp + 1, stride):
            segment = tracks[i, start:start + n_extrp]
            segments.append(segment)
    if not segments:
        raise ValueError(f"No segments extracted! Data length {length_track} < {n_extrp}")
    return torch.stack(segments)


def normalize_data(normalization, data):
    if normalization == 'none':
        return data, {'method': 'none'}
    
    elif normalization == 'zscore':
        mean = data.mean()
        std = data.std() + 1e-8
        normalized = (data - mean) / std
        return normalized, {'method': 'zscore', 'mean': mean, 'std': std}
    
    elif normalization == 'mean_abs':
        scales = torch.mean(torch.abs(data), dim=1, keepdim=True)  
        scales = torch.clamp(scales, min=1e-8)
        normalized = data / scales
        return normalized, {'method': 'mean_abs', 'scales': scales.squeeze()} 
    
    else:
        raise ValueError(f"Unknown normalization: {normalization}")


def train_model(model, segments, n_past, n_future, epochs, learning_rate,
                use_scheduler, scheduler_patience, scheduler_factor,
                grad_clip, device, output_dir, save_interval, batch_size=0,
                resume_checkpoint=None, weight_decay=1e-5,
                val_fraction=0.0, early_stopping_patience=0):
    """
    Train the conditional normalizing flow model.
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
        if 'loss_history' in checkpoint:
            loss_list = checkpoint['loss_history']
        if 'val_loss_history' in checkpoint:
            val_loss_list = checkpoint['val_loss_history']
        print(f"  Resuming from epoch {start_epoch}")

    segments = segments.to(device)

    # Train/validation split
    n_total = segments.shape[0]
    if val_fraction > 0.0:
        n_val = max(1, int(n_total * val_fraction))
        perm_all = torch.randperm(n_total, device=device)
        val_idx = perm_all[:n_val]
        train_idx = perm_all[n_val:]
        train_segments = segments[train_idx]
        val_segments = segments[val_idx]
    else:
        train_segments = segments
        val_segments = None

    context = train_segments[:, :n_past]
    samples = train_segments[:, n_past:n_extrp]
    n_segments = train_segments.shape[0]
    use_batches = batch_size > 0 and batch_size < n_segments

    if val_segments is not None:
        val_context = val_segments[:, :n_past]
        val_samples = val_segments[:, n_past:n_extrp]
        n_val_segments = val_segments.shape[0]
        use_val_batches = batch_size > 0 and batch_size < n_val_segments

    print(f"\nStarting training:")
    print(f"  Train segments: {n_segments}")
    if val_segments is not None:
        print(f"  Val segments:   {n_val_segments}")
    print(f"  Batch size: {batch_size if use_batches else 'full'}")
    print(f"  Epochs: {epochs}")

    for epoch in tqdm(range(start_epoch, epochs), desc="Training"):
        if use_batches:
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

        # Validation pass
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
                best_val_loss = avg_val_loss
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve_count = 0
            else:
                no_improve_count += 1

            scheduler_signal = avg_val_loss
        else:
            scheduler_signal = avg_loss

        if use_scheduler:
            scheduler.step(scheduler_signal)

        if (early_stopping_patience > 0 and val_segments is not None
                and no_improve_count >= early_stopping_patience):
            print(f"\nEarly stopping at epoch {epoch+1} "
                  f"(no val improvement for {no_improve_count} epochs)")
            break

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
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = os.path.join(output_dir, f'{args.model_name}_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to: {model_path}")

    norm_path = None
    if norm_stats is not None:
        norm_path = os.path.join(output_dir, f'norm_stats_{timestamp}.npz')
        np.savez(norm_path,
                 mean=norm_stats['mean'].cpu().numpy(),
                 std=norm_stats['std'].cpu().numpy())
        print(f"Saved norm stats to: {norm_path}")

    loss_path = os.path.join(output_dir, f'loss_history_{timestamp}.npy')
    np.save(loss_path, np.array(loss_list))
    if val_loss_list:
        val_loss_path = os.path.join(output_dir, f'val_loss_history_{timestamp}.npy')
        np.save(val_loss_path, np.array(val_loss_list))

    config = vars(args).copy()
    config['timestamp'] = timestamp
    config['final_loss'] = loss_list[-1] if loss_list else None
    config['final_val_loss'] = val_loss_list[-1] if val_loss_list else None
    if extra_config:
        config.update(extra_config)
    config_path = os.path.join(output_dir, f'config_{timestamp}.json')
    config_path = os.path.join(output_dir, f'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(loss_list, label='Train Loss')
    if val_loss_list:
        plt.plot(val_loss_list, label='Val Loss', linestyle='--')
    title = 'Training vs Validation Loss'
    if landscape_name:
        title = f"{landscape_name} — {title}"
    plt.title(title, fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Negative Log Likelihood', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(output_dir, f'training_loss_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved loss plot to: {plot_path}")

    return model_path, norm_path   


def plot_raw_data_sample(data, output_dir, timestamp, n_plots=9,
                         well_positions=None, landscape_name=None):
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
        if well_positions:
            for j, wp in enumerate(well_positions):
                ax.axhline(y=wp, color='gray', linestyle='--', linewidth=1.2,
                           alpha=0.7, zorder=0,
                           label=f"Well {j + 1}" if i == 0 else '_nolegend_')
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


def visualize_predictions(model, data, norm_stats, n_past, n_future, device,
                          output_dir, timestamp, well_positions=None,
                          landscape_name=None, plot_future_steps=None):
    print("\nGenerating prediction visualizations...")
    if norm_stats is not None:
        norm_data = (data - norm_stats['mean']) / norm_stats['std']
        mu_np = norm_stats['mean'].cpu().numpy()
        std_np = norm_stats['std'].cpu().numpy()
    else:
        norm_data = data
        mu_np = 0
        std_np = 1
    pfs = min(plot_future_steps, n_future) if plot_future_steps is not None else n_future
    y_label = (r'angle $\varphi$ (rad)'
               if landscape_name and 'alanine' in landscape_name.lower()
               else r'Position, $x$')
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
        samples_norm = None
        start = 0
        for attempt in range(3):
            start = np.random.randint(0, max(1, traj_len - n_extrp + 1))
            past_norm = norm_data[idx, start:start + n_past].unsqueeze(0).to(device)
            past_repeat = past_norm.repeat(500, 1)
            with torch.no_grad():
                try:
                    samples_norm = model.sample(500, past_repeat)[0].cpu().numpy()
                    break
                except AssertionError as e:
                    print(f"  WARNING: Sampling failed for traj {idx} start={start}: {e}")
        if samples_norm is None:
            ax.set_title(f"Trajectory {idx} [FAILED]")
            continue
        samples_real = (samples_norm * std_np) + mu_np
        real_traj = data[idx, start:start + n_extrp].cpu().numpy()
        future_steps = np.arange(n_past, n_past + pfs)
        all_steps = np.arange(0, n_past + pfs)
        median_disp = np.median(samples_real[:, :pfs], axis=0)
        lower_disp = np.percentile(samples_real[:, :pfs], 2.5, axis=0)
        upper_disp = np.percentile(samples_real[:, :pfs], 97.5, axis=0)
        lower_50_disp = np.percentile(samples_real[:, :pfs], 25, axis=0)
        upper_50_disp = np.percentile(samples_real[:, :pfs], 75, axis=0)
        if well_positions:
            for j, wp in enumerate(well_positions):
                ax.axhline(y=wp, color='gray', linestyle='--', linewidth=1.2,
                           alpha=0.7, zorder=0,
                           label=f"Well {j + 1}" if i == 0 else '_nolegend_')
        ax.plot(all_steps, real_traj[:n_past + pfs], 'k-', linewidth=1.5, label='Truth')
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

    landscape_name = args.landscape
    if landscape_name is None:
        landscape_name = os.path.splitext(os.path.basename(args.data_path))[0]

    plot_future_steps = args.plot_future_steps if args.plot_future_steps is not None else args.n_future

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    device = setup_device(args.device)

    reshaped_data = load_data(args.data_path)

    norm_stats = None
    if args.normalization != 'none':
        norm_data, norm_stats = normalize_data(args.normalization, reshaped_data)
        if args.normalization == 'zscore':
            print(f"Z‑score normalization: mean={norm_stats['mean']:.4f}, std={norm_stats['std']:.4f}")
        elif args.normalization == 'mean_abs':
            print(f"Mean‑abs normalization: scales range [{norm_stats['scales'].min():.4f}, {norm_stats['scales'].max():.4f}]")
    else:
        norm_data = reshaped_data

    segments = extract_segments(norm_data, args.n_past, args.n_future, args.stride)
    print(f"Extracted {segments.shape[0]} segments of length {segments.shape[1]}")

    # Plot raw data sample
    try:
        timestamp_data = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_raw_data_sample(reshaped_data, args.output_dir, timestamp_data,
                             well_positions=args.well_positions,
                             landscape_name=landscape_name)
    except Exception as e:
        print(f"WARNING: Raw data visualization failed: {e}")

    # Crete model
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
        model = preset_stage1(device, n_past=args.n_past, n_future=args.n_future, past_dim=1,encoder=args.encoder_type,
                            K=args.flow_blocks, hidden_units=args.hidden_units, hidden_layers_list=hidden_layers_list,)
    elif args.model_variant == "ar_encoder_light":
        model = preset_stage2(device, n_past=args.n_past, n_future=args.n_future, past_dim=1, )
    else:
        raise ValueError(f"Unknown model_variant: {args.model_variant}")

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
        val_fraction=args.val_fraction,
        early_stopping_patience=args.early_stopping_patience,
    )
    _training_time_seconds = time.time() - _train_start
    print(f"\nTotal training time: {_training_time_seconds:.1f}s ({_training_time_seconds/60:.1f} min)")

    _samp_n = 100
    _samp_ctx = segments[:1, :args.n_past].to(device)
    _samp_rep = _samp_ctx.expand(_samp_n, -1)
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
          f"{_sampling_time*1000:.1f}ms total ({_time_per_sample*1000:.3f}ms/sample)")

    _timing_extra = {
        "training_time_seconds": _training_time_seconds,
        "sampling_benchmark_n_samples": _samp_n,
        "sampling_benchmark_total_seconds": _sampling_time,
        "sampling_time_per_sample_seconds": _time_per_sample,
    }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path, norm_path = save_results(model, loss_list, norm_stats, args, args.output_dir,
                                         val_loss_list=val_loss_list,
                                         landscape_name=landscape_name,
                                         extra_config=_timing_extra)

    # Visualize predictions
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

    print("\n" + "="*50)
    print("Training complete!")
    print(f"Final train loss: {loss_list[-1]:.6f}")
    if val_loss_list:
        print(f"Final val loss:   {val_loss_list[-1]:.6f}")
    print(f"Results saved to: {args.output_dir}")
    print("="*50)


if __name__ == '__main__':
    main()