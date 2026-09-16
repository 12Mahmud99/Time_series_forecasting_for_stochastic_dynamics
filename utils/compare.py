#!/usr/bin/env python3
"""
compare.py – Generate all figures (SVG) and a summary table (PNG/CSV) with metrics including time.
Assumes samples shape (N, H, S). Produces:
  - raw_trajectories_{landscape}.svg
  - trajectory_comparison_{landscape}.svg
  - histogram2d_comparison_{landscape}_dark.svg   (no visible bin edges)
  - histogram2d_comparison_{landscape}_light.svg  (with bin edges)
  - error_metrics_{landscape}_len{length}.svg
  - summary_{landscape}_len{length}.png
  - summary_{landscape}_len{length}.csv
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict] = {
    "nftsf":      {"label": "NFTSF",       "color": "#1F77B4"},
    "nf-transformer":      {"label": "NF-T",       "color": "#EB69D9"},
    "nf-cnn":      {"label": "NF-CNN",       "color": "#8F11E2"},
    "nf-cnn_2":      {"label": "NF-CNN_2",       "color": "#8F11E3"},
    "nf-cnn_3":      {"label": "NF-CNN_3",       "color": "#8F11E4"},
    
    "nf_cnn_k3":      {"label": "NF-CNN-k3",       "color": "#1F77B4"},
    "nf_cnn_k3_2":      {"label": "NF-CNN-k3-2",       "color": "#EB69D9"},
    "nf_cnn_k3_3":      {"label": "NF-CNN-k3-3",       "color": "#8F11E2"},
    "nf_cnn_k4":      {"label": "NF-CNN-k4",       "color": "#8F11E3"},
    "nf_cnn_k4_2":      {"label": "NF-CNN-k4-2",       "color": "#8F11E4"},
    "nf_cnn_k4_3":      {"label": "NF-CNN-k4-3",       "color": "#8F11E4"},
    
    "arima":      {"label": "ARIMA",        "color": "#2CA02C"},
    "tsdiff_cond":{"label": "TSDiff-Cond",  "color": "#C71FD6"},
    "tsdiff_cond_2":{"label": "TSDiff-Cond_2",  "color": "#C71FD5"},
    "tsdiff_cond_3":{"label": "TSDiff-Cond_3",  "color": "#C71FD4"},
    "tsdiff_ms":  {"label": "TSDiff-MS",    "color": "#A09E2C"},
    "tsdiff_ms_2":  {"label": "TSDiff-MS_2",    "color": "#A09E2A"},
    "tsdiff_ms_3":  {"label": "TSDiff-MS_3",    "color": "#A09E2B"},
    "tsdiff_q":   {"label": "TSDiff-Q",     "color": "#B41F1F"},
    "tsdiff_q_2":   {"label": "TSDiff-Q_2",     "color": "#B41F1C"},
    "tsdiff_q_3":   {"label": "TSDiff-Q_3",     "color": "#B41F1A"},
    "csdi":       {"label": "CSDI",         "color": "#0AF1F1"},
    "csdi_2":       {"label": "CSDI_2",         "color": "#0AF1F2"},
    "csdi_3":       {"label": "CSDI_3",         "color": "#0AF1F3"},
}

# ---------------------------------------------------------------------------
# Landscape display helpers
# ---------------------------------------------------------------------------
LANDSCAPE_DISPLAY: dict[str, str] = {
    "single_well": "Single Well",
    "double_well": "Double Well",
    "alanine_phi": r"Alanine $\varphi$ (Phi)",
    "alanine_psi": r"Alanine $\psi$ (Psi)",
}
COORDINATE_LABEL: dict[str, str] = {
    "single_well": r"Single Well",
    "double_well": r"Double Well",
    "memory_kernel": r"Memory Kernel",
    "alanine_phi": r"Angle $\varphi$ (rad)",
    "alanine_psi": r"Angle $\psi$ (rad)",
}

def _land_display(landscape: str) -> str:
    return LANDSCAPE_DISPLAY.get(landscape, landscape.replace("_", " ").title())

def _coord_label(landscape: str) -> str:
    return COORDINATE_LABEL.get(landscape, r"Value")

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True,
                   help="Format: LANDSCAPE:MODEL:NPZ_PATH")
    p.add_argument("--output_dir", default="./comparison")
    p.add_argument("--n_traj_show", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--context_length", "-cl", type=int, default=0,
                   help="Fallback context length if not stored in .npz")
    p.add_argument("--data_npz", nargs="+", required=True,
                   metavar="LANDSCAPE:PATH", help="Normalization stats")
    return p.parse_args()

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------
def _load_mean_std(specs):
    result = {}
    for spec in specs:
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError(f"data_npz must be LANDSCAPE:PATH, got {spec}")
        land, path = parts
        d = np.load(path)
        result[land] = (float(d["mean"]), float(d["std"]))
        print(f"  [{land}] norm stats: mean={result[land][0]:.6f}  std={result[land][1]:.6f}")
    return result

def _denorm(x, mean, std):
    return x * (std + 1e-8) + mean

# ---------------------------------------------------------------------------
# Result loading – assumes samples shape (N, H, S) and reads time_elapsed
# ---------------------------------------------------------------------------
def load_npz_result(npz_path: str, context_length_fallback: int, mean=None, std=None):
    data = np.load(npz_path, allow_pickle=True)
    samples = data["samples"]                     # (N, H, S)
    full_trajectories = data["full_trajectories"]
    try:
        train_test_split = int(data["train_test_split"])
    except:
        train_test_split = 900    
    H = int(data["prediction_length"])
    N = samples.shape[0]
    if samples.ndim != 3 or samples.shape[1] != H:
        raise ValueError(f"Expected (N, H, S) with H={H}, got {samples.shape}")
    S = samples.shape[2]
    try:
        L = int(data["context_length"])
    except (KeyError, ValueError):
        L = context_length_fallback
    start_idx = train_test_split - L
    end_idx   = train_test_split + H
    if start_idx < 0 or end_idx > full_trajectories.shape[1]:
        print(f"WARNING: adjusting window for {npz_path}")
        end_idx = full_trajectories.shape[1]
        start_idx = end_idx - (L + H)
        if start_idx < 0:
            L = full_trajectories.shape[1] - H
            start_idx = 0
    ground_truths = full_trajectories[:, start_idx:end_idx]   # (N, L+H)
    if mean is not None and std is not None:
        ground_truths = _denorm(ground_truths, mean, std)
        samples = _denorm(samples, mean, std)
    # Read time_elapsed if present
    time_elapsed = float(data["time_elapsed"]) if "time_elapsed" in data else np.nan
    return {
        "ground_truths": ground_truths.astype(np.float32),
        "samples": samples.astype(np.float32),
        "n_past": L,
        "n_future": H,
        "fullset_N": N,
        "pred_len": H,
        "time_elapsed": time_elapsed,
    }

# ---------------------------------------------------------------------------
# Metrics (all assuming samples shape (N, H, S))
# ---------------------------------------------------------------------------
def _mae_median_per_step(gt_future, samples):
    """Per-step MAE using median of samples."""
    median = np.median(samples, axis=2)
    return np.abs(gt_future - median).mean(axis=0)

def _mae_mean_per_step(gt_future, samples):
    """Per-step MAE using mean of samples."""
    mean_forecast = np.mean(samples, axis=2)
    return np.abs(gt_future - mean_forecast).mean(axis=0)

def _mae_sample_per_step(gt_future, samples):
    """Per-step MAE (sample): average absolute error over samples and trajectories."""
    return np.abs(samples - gt_future[:, :, np.newaxis]).mean(axis=(0, 2))

def _crps_ensemble_per_step(gt_future, samples):
    """Per-step CRPS (energy score)."""
    N, H = gt_future.shape
    S = samples.shape[2]
    crps_step = np.zeros(H)
    for t in range(H):
        fc_t = samples[:, t, :]           # (N, S)
        obs_t = gt_future[:, t]           # (N,)
        fc_flat = fc_t.reshape(-1, S)
        obs_flat = obs_t.reshape(-1)
        fc_sorted = np.sort(fc_flat, axis=1)
        k = np.arange(1, S+1, dtype=np.float64)
        weights = 2.0 * k - S - 1.0
        spread = (fc_sorted * weights).sum(axis=1) / (S * (S - 1))
        mae_term = np.abs(fc_flat - obs_flat[:, None]).mean(axis=1)
        crps_step[t] = np.mean(mae_term - spread)
    return crps_step

def _ci_coverage_per_step(gt_future, samples, pct):
    """Per-step central coverage for a given percentage (50 or 90)."""
    lo = (100 - pct) / 2.0
    hi = 100 - lo
    lo_q = np.percentile(samples, lo, axis=2)
    hi_q = np.percentile(samples, hi, axis=2)
    covered = (gt_future >= lo_q) & (gt_future <= hi_q)
    return np.mean(covered, axis=0)

def _isce_central_per_step(gt_future, samples):
    """Per-step ISCE (integrated squared calibration error) using 1% steps."""
    H = gt_future.shape[1]
    pcts = np.arange(0, 101, 1)
    isce_step = np.zeros(H)
    for t in range(H):
        gt_t = gt_future[:, t]
        samp_t = samples[:, t, :]
        err = 0.0
        for pct in pcts:
            if pct == 0:
                emp = 0.0
            elif pct == 100:
                emp = 1.0
            else:
                lo = (100 - pct) / 2.0
                hi = 100 - lo
                lo_q = np.percentile(samp_t, lo, axis=1)
                hi_q = np.percentile(samp_t, hi, axis=1)
                emp = np.mean((gt_t >= lo_q) & (gt_t <= hi_q))
            err += (emp - pct/100.0)**2
        isce_step[t] = err / len(pcts)
    return isce_step

def compute_metrics(ground_truths, samples, n_past, time_elapsed):
    gt_future = ground_truths[:, n_past:]   # (N, H)
    metrics = {
        "mae_median_step": _mae_median_per_step(gt_future, samples),
        "mae_mean_step":   _mae_mean_per_step(gt_future, samples),
        "mae_sample_step": _mae_sample_per_step(gt_future, samples),
        "crps_step":       _crps_ensemble_per_step(gt_future, samples),
        "ci50_step":       _ci_coverage_per_step(gt_future, samples, 50),
        "ci90_step":       _ci_coverage_per_step(gt_future, samples, 90),
        "isce_step":       _isce_central_per_step(gt_future, samples),
    }
    # Scalar metrics for table
    metrics["mae_median"] = np.mean(metrics["mae_median_step"])
    metrics["mae_mean"]   = np.mean(metrics["mae_mean_step"])
    metrics["mae_sample"] = np.mean(metrics["mae_sample_step"])
    metrics["crps"]       = np.mean(metrics["crps_step"])
    metrics["ci50"]       = np.mean(metrics["ci50_step"])
    metrics["ci90"]       = np.mean(metrics["ci90_step"])
    metrics["isce_mean"]  = np.mean(metrics["isce_step"])
    metrics["time_elapsed"] = time_elapsed
    return metrics

# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------
def _global_ylim(model_data: dict[str, dict]) -> tuple[float, float]:
    g_lo, g_hi = float("inf"), float("-inf")
    for mdata in model_data.values():
        for gt in mdata["ground_truths"]:
            g_lo = min(g_lo, gt.min())
            g_hi = max(g_hi, gt.max())
    margin = 0.05 * (g_hi - g_lo) if g_hi > g_lo else 0.1
    return g_lo - margin, g_hi + margin

# ---------------------------------------------------------------------------
# Figure E: Raw ground-truth trajectories (SVG)
# ---------------------------------------------------------------------------
def plot_raw_trajectories(landscape, full_trajectories, n_past, n_future, output_dir, seed=42):
    rng = np.random.default_rng(seed)
    data = full_trajectories
    N, T = data.shape
    n_extrp = n_past + n_future
    x_all = np.arange(n_extrp)
    counts = [10, 100, N]

    all_idx = rng.choice(N, size=N, replace=False)
    all_starts = np.array([rng.integers(0, max(1, T - n_extrp + 1)) for _ in range(N)])

    y_lo, y_hi = float("inf"), float("-inf")
    for i in range(N):
        seg = data[all_idx[i], all_starts[i]:all_starts[i] + n_extrp]
        if len(seg) == n_extrp:
            y_lo = min(y_lo, seg.min())
            y_hi = max(y_hi, seg.max())
    margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
    y_lo -= margin
    y_hi += margin

    fig, axes = plt.subplots(1, 3, figsize=(4.5, 1.125), sharey=True)
    SINGLE_COLOR = "#4C72B0"
    for ax, count in zip(axes, counts):
        n = min(count, N)
        indices = all_idx[:n]
        starts = all_starts[:n]
        if n <= 10:
            colors = plt.get_cmap("tab10")(np.linspace(0, 1, n))
            lw, alpha = 1.5, 1.0
        elif n <= 100:
            colors = [SINGLE_COLOR] * n
            lw, alpha = 0.7, 0.25
        else:
            colors = [SINGLE_COLOR] * n
            lw, alpha = 0.5, 0.10
        for i, (idx, s) in enumerate(zip(indices, starts)):
            seg = data[idx, s:s + n_extrp]
            if len(seg) < n_extrp: continue
            ax.plot(x_all, seg, color=colors[i] if n <= 10 else SINGLE_COLOR,
                    linewidth=lw, alpha=alpha)
        ax.axvline(x=n_past, color="black", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.set_xlim(0, n_extrp)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Forecast step", fontsize=5)
        ax.set_title(f"{n} Trajectories", fontsize=6)
        ax.tick_params(labelsize=4)
    axes[0].set_ylabel(_coord_label(landscape), fontsize=5)
    fig.suptitle(f"{_land_display(landscape)} — Ground Truth Trajectories", fontsize=6, y=1.02)
    plt.tight_layout()
    out_path = output_dir / f"raw_trajectories_{landscape}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[E] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure A: Trajectory comparison grid (SVG)
# ---------------------------------------------------------------------------
def plot_trajectory_comparison_grid(landscape, display_labels, model_data, n_past, n_future, output_dir):
    n_models = len(model_data)
    n_traj = len(display_labels)
    
    row_height = 1.125
    fig_width = 4.5
    fig_height = row_height * n_models
    fig, axes = plt.subplots(n_models, n_traj, figsize=(fig_width, fig_height), squeeze=False)
    
    future_steps = np.arange(n_past, n_past + n_future)
    all_steps = np.arange(0, n_past + n_future)
    
    # Global y-limits – force for wells, else compute from data
    if landscape in ["single_well", "double_well"]:
        g_lo, g_hi = -2.5, 2.5
        y_ticks = [-2, -1, 0, 1, 2]
    elif landscape in ["memory_kernel"]:
        g_lo, g_hi = -0.5, 0.5
        y_ticks = [-0.5, -0.25, 0, 0.25, 0.5]
    else:
        g_lo, g_hi = _global_ylim(model_data)
        y_ticks = None

    use_pi_format = landscape in ("alanine_phi", "alanine_psi")
    if use_pi_format:
        from matplotlib.ticker import MultipleLocator, FuncFormatter
        def pi_formatter(x, pos):
            val = round(x / (np.pi/2)) * (np.pi/2)
            if abs(val) < 1e-8: return r"$0$"
            num = val / np.pi
            if num == 1: return r"$\pi$"
            elif num == -1: return r"$-\pi$"
            elif num == 0.5: return r"$\frac{\pi}{2}$"
            elif num == -0.5: return r"$-\frac{\pi}{2}$"
            else: return f"${num:.0f}\\pi$"

    MEDIAN_COLOR = "#df1111"
    x_tick_positions = [0, n_past, n_past + n_future]
    
    # We'll collect legend handles and labels from the top‑left subplot
    legend_handles = []
    legend_labels = []
    
    for row_idx, (model_name, mdata) in enumerate(model_data.items()):
        reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
        mlabel = reg["label"]
        samples_all = mdata["samples"]
        ground_truths_all = mdata["ground_truths"]
        axes[row_idx, 0].set_ylabel(f"{mlabel}", fontsize=10)
        
        for col_idx in range(n_traj):
            ax = axes[row_idx, col_idx]
            real_traj = ground_truths_all[col_idx]
            samp = samples_all[col_idx]
            median = np.median(samp, axis=1)
            lo90 = np.percentile(samp, 5, axis=1)
            hi90 = np.percentile(samp, 95, axis=1)
            lo50 = np.percentile(samp, 25, axis=1)
            hi50 = np.percentile(samp, 75, axis=1)

            # Plot and store artists from the top‑left subplot for legend
            line_truth, = ax.plot(all_steps, real_traj, "k-", linewidth=1.0, label="Truth")
            line_median, = ax.plot(future_steps, median, color=MEDIAN_COLOR, linewidth=0.8, label="Median")
            fill50 = ax.fill_between(future_steps, lo50, hi50, color="tab:blue", alpha=0.40, label="50% band")
            fill90_low = ax.fill_between(future_steps, lo90, lo50, color="tab:orange", alpha=0.40, label="90% band")
            fill90_high = ax.fill_between(future_steps, hi50, hi90, color="tab:orange", alpha=0.40)

            if row_idx == 0 and col_idx == 0:
                from matplotlib.patches import Patch
                legend_handles = [
                    line_truth,
                    line_median,
                    Patch(facecolor="tab:blue", alpha=0.40),
                    Patch(facecolor="tab:orange", alpha=0.40)
                ]
                legend_labels = ["Truth", "Median", "50% band", "90% band"]

            ax.axvline(x=n_past, color="k", linestyle="--", alpha=0.4)
            ax.set_ylim(g_lo, g_hi)
            ax.set_xlim(0, n_past + n_future)
            
            # X-axis: only bottom row
            if row_idx == n_models - 1:
                ax.set_xlabel("Forecast step", fontsize=10)
                ax.set_xticks(x_tick_positions)
                ax.set_xticklabels(x_tick_positions, fontsize=8)
                ax.tick_params(axis='x', labelbottom=True)
            else:
                ax.tick_params(axis='x', labelbottom=False)
            
            # Y-axis: only first column
            if col_idx == 0:
                ax.tick_params(axis='y', labelsize=8, labelleft=True)
                if landscape in ["single_well", "double_well", "memory_kernel"]:
                    ax.set_yticks(y_ticks)
                elif use_pi_format:
                    ax.yaxis.set_major_locator(MultipleLocator(np.pi/2))
                    ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
            else:
                ax.tick_params(axis='y', labelleft=False)
    
    # Add a single horizontal legend at the top of the figure
    fig.legend(legend_handles, legend_labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 0.98),
               ncol=4,               # four items in one row
               fontsize=8,
               framealpha=0.8)
    
    fig.suptitle(f"{_land_display(landscape)} — Trajectory Comparison", fontsize=10, y=1.0)
    # Adjust top margin to make room for the legend
    plt.subplots_adjust(top=0.90)
    plt.tight_layout()
    out_path = output_dir / f"trajectory_comparison_{landscape}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[A] Saved: {out_path}")

# ---------------------------------------------------------------------------
# Figure B: 2-D histogram comparison grid (both dark and light backgrounds)
# ---------------------------------------------------------------------------
def plot_histogram2d_comparison_grid(landscape, display_labels, model_data, n_past, n_future, output_dir, dark=True):
    style = 'dark_background' if dark else 'default'
    with plt.style.context(style):
        n_models = len(model_data)
        n_traj = len(display_labels)
        
        row_height = 1.125
        fig_width = 4.5
        fig_height = row_height * n_models
        fig, axes = plt.subplots(n_models, n_traj, figsize=(fig_width, fig_height), squeeze=False)
        
        future_steps = np.arange(n_past, n_past + n_future)
        all_steps = np.arange(0, n_past + n_future)
        
        # Y-limits
        if landscape in ["single_well", "double_well"]:
            g_lo, g_hi = -2.2, 2.2
            y_ticks = [-2, -1, 0, 1, 2]
            
        if landscape in ["memory_kernel"]:
            g_lo, g_hi = -1.2, 1.2
            y_ticks = [-1.2, -0.6, 0, 0.6, 1.2]
        else:
            g_lo, g_hi = _global_ylim(model_data)
            y_ticks = None
        
        n_x_bins = max(12, n_future // 5)
        n_y_bins = 24
        x_edges = np.linspace(n_past, n_past + n_future, n_x_bins + 1)
        y_edges = np.linspace(g_lo, g_hi, n_y_bins + 1)
        
        use_pi_format = landscape in ("alanine_phi", "alanine_psi")
        if use_pi_format:
            from matplotlib.ticker import MultipleLocator, FuncFormatter
            def pi_formatter(x, pos):
                val = round(x / (np.pi/2)) * (np.pi/2)
                if abs(val) < 1e-8: return r"$0$"
                num = val / np.pi
                if num == 1: return r"$\pi$"
                elif num == -1: return r"$-\pi$"
                elif num == 0.5: return r"$\frac{\pi}{2}$"
                elif num == -0.5: return r"$-\frac{\pi}{2}$"
                else: return f"${num:.0f}\\pi$"
        
        label_color = 'white' if dark else 'black'
        title_color = 'white' if dark else 'black'
        facecolor = 'black' if dark else 'white'
        x_tick_positions = [0, n_past, n_past + n_future]
        
        for row_idx, (model_name, mdata) in enumerate(model_data.items()):
            reg = MODEL_REGISTRY.get(model_name, {"label": model_name, "color": "tab:orange"})
            mlabel = reg["label"]
            samples_all = mdata["samples"]
            ground_truths_all = mdata["ground_truths"]
            axes[row_idx, 0].set_ylabel(f"{mlabel}", fontsize=10, color=label_color)
            
            for col_idx in range(n_traj):
                ax = axes[row_idx, col_idx]
                real_traj = ground_truths_all[col_idx]
                samp = samples_all[col_idx]
                samp_for_hist = samp.T
                time_rep = np.tile(future_steps, (samp_for_hist.shape[0], 1))
                
                if dark:
                    ax.hist2d(time_rep.flatten(), samp_for_hist.flatten(),
                              bins=[x_edges, y_edges], cmap="magma", density=True, norm=LogNorm(vmin=1e-6),
                              edgecolors='none', rasterized=True)
                else:
                    ax.hist2d(time_rep.flatten(), samp_for_hist.flatten(),
                              bins=[x_edges, y_edges], cmap="magma", density=True, norm=LogNorm(vmin=1e-6))
                ax.plot(all_steps, real_traj, color="lime", linewidth=1.0, label="Truth")
                ax.axvline(x=n_past, color="gray", linestyle="--", alpha=0.5)
                ax.set_ylim(g_lo, g_hi)
                ax.set_xlim(0, n_past + n_future)
                ax.set_facecolor(facecolor)
                
                # X-axis: only bottom row
                if row_idx == n_models - 1:
                    ax.set_xlabel("Forecast step", fontsize=10, color=label_color)
                    ax.set_xticks(x_tick_positions)
                    ax.set_xticklabels(x_tick_positions, fontsize=8, color=label_color)
                    ax.tick_params(axis='x', labelbottom=True)
                else:
                    ax.tick_params(axis='x', labelbottom=False)
                
                # Y-axis: only first column
                if col_idx == 0:
                    ax.tick_params(axis='y', labelsize=8, labelleft=True, colors=label_color)
                    if landscape in ["single_well", "double_well", "memory_kernel"]:
                        ax.set_yticks(y_ticks)
                    elif use_pi_format:
                        ax.yaxis.set_major_locator(MultipleLocator(np.pi/2))
                        ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))
                else:
                    ax.tick_params(axis='y', labelleft=False)
                
                if row_idx == 0 and col_idx == 0:
                    leg = ax.legend(loc='upper left', fontsize=8, framealpha=0.8)
                    if dark:
                        leg.get_frame().set_facecolor('black')
                        leg.get_frame().set_edgecolor('white')
                        for text in leg.get_texts():
                            text.set_color('white')
        
        fig.suptitle(f"{_land_display(landscape)} — Prediction Density Comparison", fontsize=10, y=0.96, color=title_color)
        plt.subplots_adjust(top=0.93)
        plt.tight_layout()
        suffix = "_dark" if dark else "_light"
        out_path = output_dir / f"histogram2d_comparison_{landscape}{suffix}.svg"
        fig.savefig(out_path, format="svg", bbox_inches="tight")
        plt.close(fig)
        print(f"[B] Saved: {out_path} (background={'dark' if dark else 'light'})")
# ---------------------------------------------------------------------------
# Figure C: Error metrics grid (horizontal, 5 columns, legend only in leftmost)
# ---------------------------------------------------------------------------
def plot_error_metrics_grid(land, length, model_names, all_metrics, output_dir):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    from matplotlib.lines import Line2D

    def add_scaling_outside(ax, exponent):
        if exponent == 0:
            return
        ax.annotate(f"$\\times 10^{{{exponent}}}$",
                    xy=(0, 1.08), xycoords='axes fraction',
                    ha='left', va='bottom', fontsize=8)

    def set_metric_xticks(ax, H):
        if H == 25:
            ticks = [0, 25]
        elif H == 50:
            ticks = [0, 25, 50]
        else:
            return
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=8)

    fig = plt.figure(figsize=(5.5, 1.125*2+1))
    outer_gs = fig.add_gridspec(2, 1, hspace=1.1, height_ratios=[1, 1])
    top_gs = outer_gs[0].subgridspec(1, 4, width_ratios=[0.5, 1, 1, 0.5], wspace=0.5)
    axes = {
        "mae": fig.add_subplot(top_gs[0, 1]),
        "crps": fig.add_subplot(top_gs[0, 2]),
    }
    bottom_gs = outer_gs[1].subgridspec(1, 3, wspace=0.5)
    axes["ci50"] = fig.add_subplot(bottom_gs[0, 0])
    axes["ci90"] = fig.add_subplot(bottom_gs[0, 1])
    axes["isce"] = fig.add_subplot(bottom_gs[0, 2])

    metric_configs = [
        ("mae_sample_step", "MAE", "mae", 10, -1),
        ("crps_step", "CRPS", "crps", 10, -1),
        ("ci50_step", r"$CI_{.50}$", "ci50", 1, -1),
        ("ci90_step", r"$CI_{.90}$", "ci90", 1, -1),
        ("isce_step", "ISCE", "isce", 1000, -3),
    ]

    legend_handles = []
    legend_labels = []

    for key, title, ax_key, multiplier, ann_exp in metric_configs:
        ax = axes[ax_key]

        # Collect raw values and apply multiplier
        all_vals = np.concatenate([all_metrics[m][key] for m in model_names]) * multiplier
        # For CI, we'll also need the raw original values for the ideal line? The multiplier for CI is 1, so fine.

        # Plot each model (also multiplied)
        for mn in model_names:
            color = MODEL_REGISTRY[mn]["color"]
            vals = all_metrics[mn][key] * multiplier
            steps = np.arange(1, len(vals) + 1)
            line, = ax.plot(steps, vals, color=color, linewidth=0.8)
            if ax_key == "mae":
                legend_handles.append(Line2D([0], [0], color=color, lw=1.5))
                legend_labels.append(MODEL_REGISTRY[mn]["label"])

        # Ideal line for CI (ideal value also multiplied: 0.5*multiplier or 0.9*multiplier)
        if "CI" in title:
            ideal = 0.5 if ".50" in title else 0.9
            ideal_multiplied = ideal * multiplier
            ax.axhline(ideal_multiplied, color="black", linestyle="--", linewidth=0.5)

        ax.set_xlabel("Forecast step", fontsize=10)
        ax.set_ylabel(title, fontsize=10)

        min_val = np.min(all_vals)
        max_val = np.max(all_vals)

        if "CI" in title:
            min_val = np.min(all_vals)
            max_val = np.max(all_vals)
            min_val=min_val-0.05
            max_val=max_val+0.05
            ideal_val = 0.5 * multiplier if ".50" in title else 0.9 * multiplier
            ticks = [min_val, ideal_val, max_val]
            # Min label: original value ×10 (rounded)
            if multiplier == 1:
                min_label = int(round(min_val * 10))
            else:
                min_label = int(round(min_val * 10 / multiplier))
            if ".50" in title:
                labels = [f"{min_label}", "5", ""]
            else:
                labels = [f"{min_label}", "9", ""]
            # Set tight y‑limits and fixed ticks
            ax.set_ylim(min_val, max_val)
            ax.yaxis.set_major_locator(ticker.FixedLocator(ticks))
            ax.set_yticklabels(labels, fontsize=8)
            # Reapply limits to disable any auto‑margin
            ax.set_ylim(min_val, max_val)
            # Remove any additional margin
            ax.margins(y=0)
        else:
            # MAE, CRPS, ISCE: two integer ticks at rounded min and max
            min_val = np.min(all_vals)
            max_val = np.max(all_vals)
            min_val=min_val-0.05
            max_val=max_val+0.05
            min_tick = int(np.floor(min_val))
            max_tick = int(np.ceil(max_val))
            # Ensure they are not equal (if data range is very narrow)
            if min_tick == max_tick:
                max_tick = min_tick + 1
            ax.set_ylim(min_tick, max_tick)
            ax.set_yticks([min_tick, max_tick])
            ax.set_yticklabels([f"{min_tick}", f"{max_tick}"], fontsize=8)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))

        # Add scaling annotation above the subplot
        add_scaling_outside(ax, ann_exp)

        # X‑axis
        ax.set_xlim(0, length)
        set_metric_xticks(ax, length)

    # Legend
    unique_handles, unique_labels = [], []
    for h, l in zip(legend_handles, legend_labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    fig.legend(unique_handles, unique_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.14),
               ncol=len(model_names), fontsize=8, framealpha=0.8)

    plt.subplots_adjust(left=0.0, right=0.98, bottom=0.18, top=0.98,
                        wspace=0.7, hspace=0.75)
    out_path = output_dir / f"error_metrics_{land}_len{length}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[C] Saved: {out_path}")
# ---------------------------------------------------------------------------
# Figure D: Summary table (PNG) with all requested metrics
# ---------------------------------------------------------------------------
def plot_summary_table(land, length, model_names, all_metrics, output_dir):
    col_headers = [
        "Model", "MAE (med)", "MAE (mean)", "MAE (sample)", "CRPS",
        "CI50", "CI90", "ISCE (*1000)", "Time (s)"
    ]
    rows = []
    numeric_vals = []
    for mn in model_names:
        label = MODEL_REGISTRY.get(mn, {"label": mn})["label"]
        mets = all_metrics[mn]
        rows.append([
            label,
            f"{mets['mae_median']:.4f}",
            f"{mets['mae_mean']:.4f}",
            f"{mets['mae_sample']:.4f}",
            f"{mets['crps']:.4f}",
            f"{mets['ci50']:.3f}",
            f"{mets['ci90']:.3f}",
            f"{mets['isce_mean']*1000:.4f}",
            f"{mets['time_elapsed']:.2f}" if not np.isnan(mets['time_elapsed']) else "nan"
        ])
        numeric_vals.append([
            mets['mae_median'], mets['mae_mean'], mets['mae_sample'], mets['crps'],
            mets['ci50'], mets['ci90'], mets['isce_mean']*1000, mets['time_elapsed']
        ])
    # Determine best indices per column
    best_idx = []
    for col in range(8):
        col_vals = [row[col] for row in numeric_vals]
        if col in [0,1,2,3,6,7]:
            best = int(np.argmin(col_vals))
        elif col == 4:
            best = int(np.argmin(np.abs(np.array(col_vals) - 0.5)))
        elif col == 5:
            best = int(np.argmin(np.abs(np.array(col_vals) - 0.9)))
        else:
            best = None
        best_idx.append(best)
    fig, ax = plt.subplots(figsize=(12, 2 + len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.6)
    for (r,c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#D0D8E8")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#F7F9FC")
            if c >= 1 and best_idx[c-1] == r-1:
                cell.set_facecolor("#90EE90")
                cell.set_text_props(weight="bold")
    ax.set_title(f"{_land_display(land)} (L/H={length}) — Summary Metrics", fontsize=16, pad=14)
    png_path = output_dir / f"summary_{land}_len{length}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[D] Saved: {png_path}")
    csv_path = output_dir / f"summary_{land}_len{length}.csv"
    with open(csv_path, "w") as f:
        f.write(",".join(col_headers) + "\n")
        for row in rows:
            f.write(",".join(row) + "\n")
    print(f"CSV saved: {csv_path}")
    
# ---------------------------------------------------------------------------
# Figure F: Combined metrics grid (2 rows, 4 cols)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Figure F: Combined metrics grid (2 rows, 4 cols)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Figure F: Combined metrics grid (2 rows, 4 cols)
# ---------------------------------------------------------------------------# ---------------------------------------------------------------------------
# Figure F: Combined metrics grid (2 rows, 4 cols)
# ---------------------------------------------------------------------------
def plot_combined_metrics_grid(land, length, model_names, all_metrics,
                               display_labels, model_data, n_past, n_future,
                               output_dir):

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import matplotlib.ticker as ticker
    import math
    import numpy as np
    from matplotlib.lines import Line2D

    # -------------------------
    # Helpers (unchanged)
    # -------------------------
    def scale_values(vals):
        vmax = np.max(np.abs(vals))
        if vmax == 0:
            return vals, 0
        exp = int(math.floor(math.log10(vmax)))
        if exp == 0 or exp == 1:
            return vals, 0
        scale = 10 ** (-exp)
        return vals * scale, exp

    def add_scaling_outside(ax, exponent):
        if exponent == 0:
            return
        ax.annotate(f"$\\times 10^{{{exponent}}}$",
                    xy=(0, 1.08),
                    xycoords='axes fraction',
                    ha='left', va='bottom',
                    fontsize=8)

    def set_integer_ticks(ax):
        y_min, y_max = ax.get_ylim()
        n_ticks = 5
        raw_step = (y_max - y_min) / n_ticks
        if raw_step < 1:
            step = 1
        else:
            step = 10 ** np.floor(np.log10(raw_step))
            if raw_step / step < 2:
                step = step
            elif raw_step / step < 5:
                step = 2 * step
            else:
                step = 5 * step
            step = int(step)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(step))
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)

    def set_metric_xticks(ax, H):
        if H == 25:
            ticks = [0, 25]
        elif H == 50:
            ticks = [0, 25, 50]
        else:
            return
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, fontsize=8)

    # -------------------------
    # Figure layout
    # -------------------------
    first_model = model_names[0]
    traj_indices = [0, 1, 2]

    fig, axes = plt.subplots(2, 4, figsize=(5.5, 2.25), squeeze=False)

    # -------------------------
    # Y limits (wells & alanine)
    # -------------------------
    if land in ["single_well", "double_well"]:
        g_lo, g_hi = -2.2, 2.2
        y_ticks = [-2, -1, 0, 1, 2]
        use_pi_format = False
    else:
        g_lo, g_hi = _global_ylim(model_data)
        y_ticks = None
        use_pi_format = land in ("alanine_phi", "alanine_psi")
    
    if use_pi_format:
        from matplotlib.ticker import MultipleLocator, FuncFormatter
        def pi_formatter(x, pos):
            val = round(x / (np.pi/2)) * (np.pi/2)
            if abs(val) < 1e-8: return r"$0$"
            num = val / np.pi
            if num == 1: return r"$\pi$"
            elif num == -1: return r"$-\pi$"
            elif num == 0.5: return r"$\frac{\pi}{2}$"
            elif num == -0.5: return r"$-\frac{\pi}{2}$"
            else: return f"${num:.0f}\\pi$"

    future_steps = np.arange(n_past, n_past + n_future)
    all_steps = np.arange(0, n_past + n_future)

    n_x_bins = max(12, n_future // 5)
    x_edges = np.linspace(n_past, n_past + n_future, n_x_bins + 1)

    n_y_bins = 24
    y_edges = np.linspace(g_lo, g_hi, n_y_bins + 1)

    # -------------------------
    # TOP ROW: Histograms
    # -------------------------
    for col, traj_idx in enumerate(traj_indices):
        ax = axes[0, col]

        mdata = model_data[first_model]
        samples_all = mdata["samples"]
        real_traj = mdata["ground_truths"][traj_idx]

        samp = samples_all[traj_idx]
        samp_for_hist = samp.T
        time_rep = np.tile(future_steps, (samp_for_hist.shape[0], 1))

        ax.hist2d(time_rep.flatten(), samp_for_hist.flatten(),
                  bins=[x_edges, y_edges],
                  cmap="magma",
                  density=True,
                  norm=LogNorm(vmin=1e-6),
                  edgecolors='none',
                  rasterized=True)

        ax.plot(all_steps, real_traj, color="lime", linewidth=1.0)
        ax.axvline(x=n_past, color="gray", linestyle="--", alpha=0.5)

        ax.set_ylim(g_lo, g_hi)
        ax.set_xlim(0, n_past + n_future)

        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        elif use_pi_format:
            ax.yaxis.set_major_locator(MultipleLocator(np.pi/2))
            ax.yaxis.set_major_formatter(FuncFormatter(pi_formatter))

        # Y-label: landscape name (already using symbols)
        if col == 0:
            if land == "alanine_phi":
                ax.set_ylabel(r"Alanine $\varphi$", fontsize=10)
            elif land == "alanine_psi":
                ax.set_ylabel(r"Alanine $\psi$", fontsize=10)
            elif land == "single_well":
                ax.set_ylabel(r"Single Well", fontsize=10)
            elif land == "double_well":
                ax.set_ylabel(r"Double Well", fontsize=10)

        ax.set_xticks([0, n_past, n_past + n_future])
        ax.tick_params(labelsize=8)

    # -------------------------
    # TOP RIGHT: MAE (unchanged)
    # -------------------------
    ax_mae = axes[0, 3]
    all_mae_vals = np.concatenate([all_metrics[m]["mae_sample_step"] for m in model_names])
    _, exp_mae = scale_values(all_mae_vals)
    for mn in model_names:
        color = MODEL_REGISTRY[mn]["color"]
        vals = all_metrics[mn]["mae_sample_step"]
        scaled_vals, _ = scale_values(vals)
        steps = np.arange(1, len(vals) + 1)
        ax_mae.plot(steps, scaled_vals, color=color, linewidth=0.8)
    ax_mae.set_ylabel("MAE", fontsize=10)
    set_integer_ticks(ax_mae)
    add_scaling_outside(ax_mae, exp_mae)
    ax_mae.set_xlim(0, n_future)
    set_metric_xticks(ax_mae, n_future)

    # -------------------------
    # BOTTOM ROW: Metrics (unchanged)
    # -------------------------
    metric_keys = [
        ("crps_step", "CRPS"),
        ("ci50_step", r"$CI_{.50}$"),
        ("ci90_step", r"$CI_{.90}$"),
        ("isce_step", "ISCE"),
    ]

    for col, (key, title) in enumerate(metric_keys):
        ax = axes[1, col]

        all_vals = np.concatenate([all_metrics[m][key] for m in model_names])
        if key == "isce_step":
            all_vals = all_vals * 1000
        _, exp = scale_values(all_vals)
        scale = 10 ** (-exp) if exp != 0 else 1.0

        for mn in model_names:
            color = MODEL_REGISTRY[mn]["color"]
            vals = all_metrics[mn][key]
            if key == "isce_step":
                vals = vals * 1000
            scaled_vals, _ = scale_values(vals)
            steps = np.arange(1, len(vals) + 1)
            ax.plot(steps, scaled_vals, color=color, linewidth=0.8)

        if "CI" in title:
            ideal = 0.5 if ".50" in title else 0.9
            ax.axhline(ideal * scale, color="black", linestyle="--", linewidth=0.5)

        ax.set_xlabel("Forecast step", fontsize=10)
        ax.set_ylabel(title, fontsize=10)

        set_integer_ticks(ax)
        add_scaling_outside(ax, exp)

        if "CI" in title:
            if ".50" in title:
                raw_ticks = np.array([0.4, 0.5, 0.6])
                labels = ["4", "5", "6"]
            else:
                raw_ticks = np.array([0.8, 0.9])
                labels = ["8", "9"]
                
            ticks_scaled = raw_ticks * scale
            ax.set_yticks(ticks_scaled)
            ax.set_yticklabels(labels, fontsize=8)

        if key == "isce_step":
            y_min, y_max = ax.get_ylim()
            y_min_int = 0 #int(np.floor(y_min))
            y_max_int = int(np.ceil(y_max))
            if y_max_int - y_min_int < 2:
                mid = (y_min_int + y_max_int) // 2
                ticks = [y_min_int, mid, y_max_int]
            else:
                ticks = np.linspace(y_min_int, y_max_int, 3, dtype=int)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            add_scaling_outside(ax,-3)
            
        if key == "crps_step":
            y_min, y_max = ax.get_ylim()
            y_min_int = 0 #int(np.floor(y_min))
            y_max_int = int(np.ceil(y_max))
            if y_max_int - y_min_int < 2:
                mid = (y_min_int + y_max_int) // 2
                ticks = [y_min_int, mid, y_max_int]
            else:
                ticks = np.linspace(y_min_int, y_max_int, 3, dtype=int)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)

        ax.set_xlim(0, n_future)
        set_metric_xticks(ax, n_future)

    # -------------------------
    # LEGEND (unchanged)
    # -------------------------
    legend_handles = []
    legend_labels = []
    for mn in model_names:
        legend_handles.append(Line2D([0], [0], color=MODEL_REGISTRY[mn]["color"], lw=1.5))
        legend_labels.append(MODEL_REGISTRY[mn]["label"])

    fig.legend(legend_handles, legend_labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.12),
               ncol=len(model_names), fontsize=8, framealpha=0.8)

    plt.subplots_adjust(left=0.0, right=0.98,
                        bottom=0.18, top=0.92,
                        wspace=0.7, hspace=0.75)

    out_path = output_dir / f"combined_metrics_{land}_len{length}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[F] Saved: {out_path}")
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################
###########################################################################

def plot_individual_trajectories(landscape, full_trajectories, n_past, n_future, output_dir, n_traj=10):
    """
    Plot the first N ground-truth trajectories individually in a grid of subplots.
    Each subplot shows the full window (context + prediction) for a single trajectory.
    The trajectory index is shown as the title.
    """
    # Number of trajectories to plot
    N = full_trajectories.shape[0]
    n_traj = min(n_traj, N)

    # Determine grid size (e.g., 2 rows, 5 columns for 10 trajectories)
    n_cols = 5
    n_rows = (n_traj + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5), squeeze=False)
    fig.suptitle(f"{_land_display(landscape)} — Individual Ground Truth Trajectories", fontsize=10, y=0.98)

    # Global y‑limits (same as in raw_trajectories)
    if landscape in ["single_well", "double_well"]:
        y_lo, y_hi = -2.2, 2.2
        y_ticks = [-2, -1, 0, 1, 2]
    else:
        # Compute from data to avoid margins
        y_lo, y_hi = float("inf"), float("-inf")
        for i in range(n_traj):
            seg = full_trajectories[i, :n_past + n_future]
            y_lo = min(y_lo, seg.min())
            y_hi = max(y_hi, seg.max())
        margin = 0.05 * (y_hi - y_lo) if y_hi > y_lo else 0.1
        y_lo -= margin
        y_hi += margin
        y_ticks = None

    x_vals = np.arange(n_past + n_future)

    for idx in range(n_traj):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        traj = full_trajectories[idx, :n_past + n_future]
        ax.plot(x_vals, traj, color="black", linewidth=1.0)
        ax.axvline(x=n_past, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(0, n_past + n_future)
        ax.set_title(f"Trajectory {idx}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Step", fontsize=6)

        if landscape in ["single_well", "double_well"] and y_ticks is not None:
            ax.set_yticks(y_ticks)

        # Remove empty subplots
        if idx == n_traj - 1:
            for extra in range(idx + 1, n_rows * n_cols):
                fig.delaxes(axes[extra // n_cols, extra % n_cols])

    # Common y‑label only for leftmost column
    for col in range(n_cols):
        axes[0, col].set_ylabel(_coord_label(landscape), fontsize=6)

    plt.tight_layout()
    out_path = output_dir / f"individual_trajectories_{landscape}.svg"
    fig.savefig(out_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"[I] Saved: {out_path}")

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    print("\n=== Normalization stats ===")
    norm_stats = _load_mean_std(args.data_npz)

    specs = []
    for spec in args.results:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Expected LANDSCAPE:MODEL:NPZ_PATH, got {spec}")
        specs.append(parts)

    by_land_len = {}
    for land, model, path in specs:
        mean, std = norm_stats.get(land, (None, None))
        data = load_npz_result(path, args.context_length, mean, std)
        length = data["pred_len"]
        key = (land, length)
        by_land_len.setdefault(key, {})[model] = data

    for (land, length), models_data in by_land_len.items():
        print(f"\n=== {land}, prediction length = {length} ===")
        model_names = list(models_data.keys())
        all_metrics = {}
        for model, data in models_data.items():
            mets = compute_metrics(data["ground_truths"], data["samples"],
                                   data["n_past"], data["time_elapsed"])
            all_metrics[model] = mets
            label = MODEL_REGISTRY.get(model, {"label": model})["label"]
            time_str = f"{mets['time_elapsed']:.2f}s" if not np.isnan(mets['time_elapsed']) else "nan"
            print(f"  {label:12s}: MAE_med={mets['mae_median']:.4f}  MAE_mean={mets['mae_mean']:.4f}  "
                  f"MAE_samp={mets['mae_sample']:.4f}  CRPS={mets['crps']:.4f}  "
                  f"CI50={mets['ci50']:.3f}  CI90={mets['ci90']:.3f}  ISCE*1000={mets['isce_mean']*1000:.4f}  "
                  f"Time={time_str}")

        N_min = min(data["fullset_N"] for data in models_data.values())
        n_show = min(args.n_traj_show, N_min)
        #rng = np.random.default_rng(args.seed)
        #display_labels = rng.choice(N_min, size=n_show, replace=False).tolist()
        display_labels = list(range(n_show))

        display_data = {}
        for model, data in models_data.items():
            display_data[model] = {
                "ground_truths": data["ground_truths"][display_labels],
                "samples": data["samples"][display_labels],
                "n_past": data["n_past"],
                "n_future": data["n_future"],
            }
        first = next(iter(display_data.values()))
        n_past = first["n_past"]
        n_future = first["n_future"]

        first_model_data = next(iter(models_data.values()))
        plot_raw_trajectories(land, first_model_data["ground_truths"], n_past, n_future, out_dir, args.seed)
        plot_trajectory_comparison_grid(land, display_labels, display_data, n_past, n_future, out_dir)
        plot_histogram2d_comparison_grid(land, display_labels, display_data, n_past, n_future, out_dir, dark=True)
        plot_histogram2d_comparison_grid(land, display_labels, display_data, n_past, n_future, out_dir, dark=False)
        plot_error_metrics_grid(land, length, model_names, all_metrics, out_dir)
        plot_summary_table(land, length, model_names, all_metrics, out_dir)
        plot_combined_metrics_grid(land, length, model_names, all_metrics, display_labels, display_data, n_past, n_future,out_dir)

        # New line: plot individual ground truth trajectories
        plot_individual_trajectories(land, first_model_data["ground_truths"], n_past, n_future, out_dir, n_traj=10)

    print("\n✓ All outputs saved.")

if __name__ == "__main__":
    main()
    
