#!/usr/bin/env python3
"""
Unified NF-TSF forecast script for all model variants.
Produces .npz bundle compatible with compare.py.
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm
import time as timelib

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "NF"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))
sys.path.insert(0, PROJECT_ROOT)

try:
    from architecture import create_nfm
except ImportError:
    create_nfm = None

from architecture_encoder import create_nfm_encoder, preset_stage1, preset_stage2

torch.cuda.empty_cache()

def parse_args():
    p = argparse.ArgumentParser(description="Unified NF-TSF forecast script")
    p.add_argument("--model_path", required=True, help="Path to .pth checkpoint")
    p.add_argument("--config", required=True, help="Path to training config (JSON or YAML)")
    p.add_argument("--data_path", required=True, help="Path to test .npz file (with 'positions')")
    p.add_argument("--out", "-o", required=True, help="Output .npz path")
    p.add_argument("--n_samples", type=int, default=500, help="Number of ensemble samples")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Number of (traj x sample) pairs per forward pass")
    p.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train_test_split", type=int, default=None,
                   help="Override train_test_split if not in data")
    p.add_argument("--test_size", type=int, default=None,
                   help="Number of trajectories to evaluate (default: all)")
    p.add_argument("--model_variant", type=str, default=None,
                   choices=["ar", "ar_encoder_full", "ar_encoder_light"],
                   help="Override model_variant from config")
    p.add_argument("--encoder_type", type=str, default=None,
                   choices=["gru", "mlp", "cnn", "transformer"],
                   help="Encoder type (must match training; default: from config)")
    return p.parse_args()


def setup_device(device_arg):
    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"Device: {device}")
    return device


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def load_config(config_path):
    ext = os.path.splitext(config_path)[1].lower()
    if ext in ('.yaml', '.yml'):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            cfg = json.load(f)
    return cfg

def build_model(device, cfg, n_past, n_future, encoder_type="gru"):
    """Construct model according to config and optional override."""
    if n_past + n_future >= 150:
        model_variant = "ar_encoder_full"
    else:
        model_variant = "ar"
    #model_variant = model_variant_override if model_variant_override is not None else cfg.get("model_variant", "ar")
    print(f"Building model_variant: {model_variant}, encoder_type: {encoder_type}")

    # Read architecture parameters from config
    flow_blocks   = cfg.get("flow_blocks", 6)
    hidden_units  = cfg.get("hidden_units", 64)
    hidden_layers = cfg.get("hidden_layers", "1,2")
    tail_bound    = cfg.get("tail_bound", 30.0)
    hidden_layers_list = tuple(int(x) for x in str(hidden_layers).split(","))
    
    print(f"  flow_blocks: {flow_blocks}, hidden_units: {hidden_units}, hidden_layers: {hidden_layers_list}")

    if model_variant == "ar":
        
        if create_nfm is None:
            raise ImportError("architecture.py not found; cannot build 'ar' variant.")
        model = create_nfm(
            device=device,
            latent_size=n_future,
            context_size=n_past,
            K=flow_blocks,
            hidden_units=hidden_units,
            hidden_layers_list=hidden_layers_list,
            tail_bound=tail_bound,
        )
    elif model_variant in ("ar_encoder_full", "ar_encoder_light"):

        model = create_nfm_encoder(
            device=device,
            n_past=n_past,
            n_future=n_future,
            past_dim=1,
            encoder=encoder_type,
            encoder_hidden=128,
            encoder_layers=2,
            context_dim=64,
            K=flow_blocks,
            hidden_units=hidden_units,
            hidden_layers_list=hidden_layers_list,
            tail_bound=tail_bound,
        )        

    else:
        raise ValueError(f"Unknown model_variant: {model_variant}")

    return model


def load_test_data(data_path):
    data = np.load(data_path, allow_pickle=True)
    if isinstance(data, np.lib.npyio.NpzFile):
        if "positions" in data:
            positions = data["positions"].astype(np.float32)
        else:
            key = list(data.keys())[0]
            positions = data[key].astype(np.float32)
        time = data["time"] if "time" in data else None
        tts = int(data["train_test_split"]) if "train_test_split" in data else None
    else:
        positions = data.astype(np.float32)
        time = None
        tts = None
    print(f"Test data: {positions.shape} (N_traj, T)")
    return positions, time, tts


def run_forecast_batched(model, positions, context_length, prediction_length,
                         n_samples, train_test_split, test_size,
                         batch_size, device):
    test_size = min(test_size or positions.shape[0], positions.shape[0])
    positions = positions[:test_size]
    N = positions.shape[0]

    ctx_np = positions[:, train_test_split - context_length : train_test_split]
    gt_np  = positions[:, train_test_split : train_test_split + prediction_length]

    ctx_tensor = torch.tensor(ctx_np, dtype=torch.float32, device=device)

    samples_out = np.zeros((N, prediction_length, n_samples), dtype=np.float32)
    time_elapsed = 0.0

    n_batches = (N + batch_size - 1) // batch_size
    for b in tqdm(range(n_batches), desc="Forecasting"):
        b_start = b * batch_size
        b_end   = min(b_start + batch_size, N)
        B = b_end - b_start

        ctx_b = ctx_tensor[b_start:b_end]
        ctx_tiled = ctx_b.repeat_interleave(n_samples, dim=0)

        with torch.no_grad():
            start_t = timelib.time()
            out = model.sample(B * n_samples, ctx_tiled)
            if isinstance(out, tuple):
                samp = out[0]
            else:
                samp = out
            time_elapsed += timelib.time() - start_t

        samp_np = samp.cpu().numpy().reshape(B, n_samples, prediction_length)
        samples_out[b_start:b_end] = samp_np.transpose(0, 2, 1)

    print(f"Inference time: {time_elapsed:.2f}s  "
          f"({time_elapsed/N*1000:.1f} ms/traj, "
          f"{time_elapsed/(N*n_samples)*1000:.3f} ms/sample)")

    return samples_out, gt_np, ctx_np, time_elapsed


def main():
    args = parse_args()
    set_seed(args.seed)
    device = setup_device(args.device)

    cfg = load_config(args.config)

    n_past   = cfg.get("n_past", 100)
    n_future = cfg.get("n_future", 100)
    print(f"n_past={n_past}, n_future={n_future}")

    # Read encoder_type from config or use default
    encoder_type = cfg.get("encoder_type", "gru")
    if args.encoder_type is not None:
        encoder_type = args.encoder_type
    print(f"Encoder type: {encoder_type}")

    positions, time, tts_npz = load_test_data(args.data_path)
    N, T = positions.shape

    if args.train_test_split is not None:
        train_test_split = args.train_test_split
    elif tts_npz is not None:
        train_test_split = tts_npz
    else:
        train_test_split = n_past
    print(f"train_test_split = {train_test_split}")

    model = build_model(device, cfg, n_past, n_future,encoder_type=encoder_type)

    state = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"Loaded model from {args.model_path}")

    test_size = args.test_size if args.test_size is not None else N
    samples, ground_truth, contexts, time_elapsed = run_forecast_batched(
        model=model,
        positions=positions,
        context_length=n_past,
        prediction_length=n_future,
        n_samples=args.n_samples,
        train_test_split=train_test_split,
        test_size=test_size,
        batch_size=args.batch_size,
        device=device,
    )

    ci90_lower = np.percentile(samples,  5, axis=2)
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    out_path = args.out
    if not out_path.endswith(".npz"):
        out_path += ".npz"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    save_dict = {
        "samples": samples,
        "ground_truth": ground_truth,
        "contexts": contexts,
        "ci90_lower": ci90_lower,
        "ci90_upper": ci90_upper,
        "ci50_lower": ci50_lower,
        "ci50_upper": ci50_upper,
        "full_trajectories": positions[:test_size],
        "train_test_split": train_test_split,
        "prediction_length": n_future,
        "context_length": n_past,
        "num_of_samples": args.n_samples,
        "time_elapsed": time_elapsed,
    }
    if time is not None:
        save_dict["time"] = time
        save_dict["time_train"] = time[:train_test_split]
        save_dict["time_test"] = time[train_test_split : train_test_split + n_future]

    np.savez_compressed(out_path, **save_dict)
    print(f"\nSaved: {out_path}")
    print(f"samples : {samples.shape} (N, H, S)")
    print(f"ground_truth : {ground_truth.shape} (N, H)")
    print(f"contexts : {contexts.shape} (N, L)")
    print(f"time_elapsed : {time_elapsed:.2f}s")

    assert not np.isnan(samples).any(), "NaN in samples"
    assert not np.isinf(samples).any(), "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("All sanity checks passed.")


if __name__ == "__main__":
    main()