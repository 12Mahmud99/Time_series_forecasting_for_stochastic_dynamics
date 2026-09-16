#!/usr/bin/env python3
"""
TSDiff-Cond inference — per-batch timing + optional denormalisation.

Uses the SAME train_test_split-based windowing approach as the second script:
    context  = [train_test_split - L : train_test_split]
    forecast = [train_test_split     : train_test_split + H]
This guarantees exact alignment with the ground truth, regardless of how the
GluonTS dataset was originally built.
"""

import argparse
import logging
import yaml
import numpy as np
import torch
import time as timelib
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import MetaData, FileDataset, ListDataset, TrainDatasets
from gluonts.evaluation import make_evaluation_predictions

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    MaskInput,
)
from uncond_ts_diff.model import TSDiffCond
import uncond_ts_diff.configs as diffusion_configs

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(config: dict) -> TSDiffCond:
    """Load TSDiffCond model from checkpoint."""
    model = TSDiffCond(
        **getattr(
            diffusion_configs,
            config.get("diffusion_config", "diffusion_small_config"),
        ),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        init_skip=config["init_skip"],
        noise_observed=config.get("noise_observed", True),
        normalization=config["normalization"],
    )

    logger.info(f"Loading from {config['ckpt']} ...")
    checkpoint = torch.load(config["ckpt"], map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model = model.to(config["device"])
    model.eval()
    logger.info(f"Loaded checkpoint: {config['ckpt']}")
    return model


def make_windowed_dataset(
    full_trajectories: np.ndarray,
    train_test_split: int,
    context_length: int,
    prediction_length: int,
    freq: str,
) -> ListDataset:
    """
    Build an in-memory GluonTS ListDataset where each series is truncated to
    exactly [train_test_split - context_length : train_test_split + prediction_length].
    """
    start_idx = train_test_split - context_length
    end_idx = train_test_split + prediction_length

    assert start_idx >= 0, \
        f"train_test_split ({train_test_split}) < context_length ({context_length})"
    assert end_idx <= full_trajectories.shape[1], \
        (f"train_test_split ({train_test_split}) + prediction_length ({prediction_length})"
         f" = {end_idx} > T ({full_trajectories.shape[1]})")

    entries = [
        {
            FieldName.TARGET: traj[start_idx:end_idx].astype(np.float32),
            FieldName.START: pd.Timestamp("2000-01-01"),
            FieldName.ITEM_ID: str(i),
        }
        for i, traj in enumerate(full_trajectories)
    ]

    logger.info(
        f"Built ListDataset: {len(entries)} series "
        f"of length {context_length + prediction_length}  "
        f"(steps {start_idx}–{end_idx-1})"
    )
    return ListDataset(entries, freq=freq)


def main():
    parser = argparse.ArgumentParser(
        description="TSDiff-Cond inference — per-batch timing + optional denormalisation."
    )
    parser.add_argument("--config", "-c", required=True,
                        help="Path to yaml config")
    parser.add_argument("--checkpoint", required=False,
                        help="Path to checkpoint (overrides ckpt in config)")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to GluonTS dataset directory")
    parser.add_argument("--out", "-o", default="results/tsdiff_cond_eval.npz",
                        help="Output .npz file path")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--num_samples", type=int,
                        help="Number of samples per forecast")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for predictor")
    parser.add_argument("--max_traj", type=int, default=None,
                        help="Limit number of test trajectories")
    parser.add_argument("--denormalize", action="store_true",
                        help="Apply denormalisation to samples (raw scale)")
    parser.add_argument("--train_test_split", "-tts", type=int, default=None,
                        help="Override train_test_split from time.npz")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if args.checkpoint:
        config["ckpt"] = args.checkpoint
    config["device"] = args.device
    if args.num_samples:
        config["num_samples"] = args.num_samples
    if args.batch_size:
        config["batch_size"] = args.batch_size

    L = config["context_length"]
    H = config["prediction_length"]
    norm_method = config.get("normalization", "none")
    logger.info(f"Context length: {L}, Prediction length: {H}")
    logger.info(f"Normalization method in config: {norm_method}")
    logger.info(f"Denormalize flag: {args.denormalize}")

    dataset_path = Path(args.dataset_path)
    with open(dataset_path / "metadata.json", "r") as f:
        meta_json = yaml.safe_load(f)
    freq = meta_json["freq"]

    # ---------- Load FULL test trajectories (all T steps) ----------
    metadata = MetaData(freq=freq, prediction_length=H)
    test_ds = FileDataset(dataset_path / "test", freq=freq)
    train_ds = FileDataset(dataset_path / "train", freq=freq)
    dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

    test_trajectories = np.array([entry["target"] for entry in dataset.test])
    N, T = test_trajectories.shape
    logger.info(f"Full test trajectories: {test_trajectories.shape}")

    # ---------- Determine train_test_split ----------
    time_npz = np.load(dataset_path / "time.npz")
    time = time_npz["time"]

    if args.train_test_split is not None:
        train_test_split = int(args.train_test_split)
        logger.info(f"Using custom train_test_split = {train_test_split}")
    else:
        train_test_split = int(time_npz["train_test_split"])
    logger.info(f"train_test_split = {train_test_split}")
    logger.info(f"Context window   : steps {train_test_split-L}–{train_test_split-1}")
    logger.info(f"Forecast window  : steps {train_test_split}–{train_test_split+H-1}")

    # Verify windows fit within the data
    assert train_test_split - L >= 0, \
        f"Context window starts at {train_test_split-L} which is before the start of data"
    assert train_test_split + H <= T, \
        f"Forecast window ends at {train_test_split+H} which exceeds T={T}"

    # Optional: limit number of trajectories
    if args.max_traj is not None:
        test_trajectories = test_trajectories[:args.max_traj]
        N = len(test_trajectories)
        logger.info(f"Limiting to {N} trajectories (--max_traj)")

    # ---------- Build windowed dataset ----------
    windowed_ds = make_windowed_dataset(
        full_trajectories=test_trajectories,
        train_test_split=train_test_split,
        context_length=L,
        prediction_length=H,
        freq=freq,
    )

    # ---------- Load model ----------
    model = load_model(config)
    lag_pad = max(model.lags_seq) if len(model.lags_seq) > 0 else 0
    logger.info(f"max(model.lags_seq) = {lag_pad}")

    # ---------- Transformation ----------
    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=H,
    )
    transformed_test = transformation.apply(windowed_ds, is_train=False)

    # ---------- Test splitter ----------
    test_splitter = create_splitter(
        past_length=L + lag_pad,
        future_length=H,
        mode="test",
    )
    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        L,
        "none",
        0,
    )
    test_transform = test_splitter + masking_transform

    # ---------- Sanity check ----------
    debug_batch = next(iter(test_transform.apply(transformed_test, is_train=False)))
    logger.info(f"past_target shape: {debug_batch['past_target'].shape}")
    logger.info(f"past_target last 10: {debug_batch['past_target'][-10:]}")
    logger.info(f"Batch size used : {config['batch_size']}")

    # ---------- Predictor ----------
    predictor = model.get_predictor(
        test_transform,
        batch_size=config["batch_size"],
        device=config["device"],
    )

    # ---------- Run forecast with per-batch timing ----------
    logger.info("Running make_evaluation_predictions...")
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_test,
        predictor=predictor,
        num_samples=config["num_samples"],
    )
    total_windows = len(list(transformed_test))

    if "cuda" in str(config["device"]):
        torch.cuda.synchronize()

    batch_size = config["batch_size"]
    per_item_times = []
    forecasts = []
    it = iter(forecast_it)
    batch_start = timelib.time()

    for idx in tqdm(range(total_windows), desc="Forecasting"):
        item = next(it)
        forecasts.append(item)
        is_batch_boundary = (idx + 1) % batch_size == 0
        is_last_item = (idx + 1) == total_windows
        if is_batch_boundary or is_last_item:
            if "cuda" in str(config["device"]):
                torch.cuda.synchronize()
            batch_elapsed = timelib.time() - batch_start
            n_in_batch = (idx + 1) - len(per_item_times)
            per_item_times.extend([batch_elapsed / n_in_batch] * n_in_batch)
            batch_start = timelib.time()

    time_elapsed = sum(per_item_times)
    logger.info(f"Inference took {time_elapsed:.2f}s total")
    logger.info(f"Per-batch avg time per trajectory: min={min(per_item_times):.3f}s, "
                f"max={max(per_item_times):.3f}s, mean={np.mean(per_item_times):.3f}s")

    # ---------- Extract samples ----------
    samples = np.array([f.samples.T for f in forecasts])  # (N, H, S)
    N_windows = samples.shape[0]

    # ---------- Ground truth and contexts from full trajectories ----------
    ground_truth = test_trajectories[
        :N_windows,
        train_test_split : train_test_split + H
    ]  # (N, H)

    contexts = test_trajectories[
        :N_windows,
        train_test_split - L : train_test_split
    ]  # (N, L)

    # Verify alignment
    logger.info("Verifying ground truth alignment...")
    assert np.allclose(
        ground_truth,
        test_trajectories[:N_windows, train_test_split:train_test_split + H],
        atol=1e-4,
    ), "Ground truth mismatch — check train_test_split and prediction_length"
    logger.info("Ground truth alignment verified ✓")

    # ---- Optional denormalisation ----
    if args.denormalize:
        if norm_method == "mean_abs":
            scales = np.mean(np.abs(contexts), axis=1, keepdims=True)  # (N, 1)
            scales = np.clip(scales, 1e-10, None)
            samples = samples * scales[:, np.newaxis, :]
            logger.info("Denormalised with mean_abs (context scale).")
        elif norm_method == "zscore":
            if "mean" in meta_json and "std" in meta_json:
                mean = meta_json["mean"]
                std = meta_json["std"]
                samples = samples * std + mean
                logger.info(f"Denormalised with zscore: mean={mean}, std={std}")
            else:
                logger.error("zscore requested but mean/std not in metadata.json.")
                raise ValueError("zscore normalisation requires mean/std in metadata.json.")
        else:
            logger.warning("Normalization method is 'none' – nothing to denormalise.")
    else:
        logger.info("Samples are saved as model output (normalised).")

    # ---- Confidence intervals ----
    ci90_lower = np.percentile(samples, 5, axis=2)
    ci90_upper = np.percentile(samples, 95, axis=2)
    ci50_lower = np.percentile(samples, 25, axis=2)
    ci50_upper = np.percentile(samples, 75, axis=2)

    # ---- Time arrays ----
    time_test = time[train_test_split : train_test_split + H]
    time_train = time[:train_test_split]

    # ---- Save ----
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "samples": samples,
        "ground_truth": ground_truth,
        "contexts": contexts,
        "full_trajectories": test_trajectories[:N_windows],
        "ci90_lower": ci90_lower,
        "ci90_upper": ci90_upper,
        "ci50_lower": ci50_lower,
        "ci50_upper": ci50_upper,
        "time": time,
        "time_train": time_train,
        "time_test": time_test,
        "train_test_split": train_test_split,
        "context_length": L,
        "prediction_length": H,
        "num_samples": config["num_samples"],
        "time_elapsed": time_elapsed,
        "per_item_times": np.array(per_item_times),
        "normalization_applied": norm_method,
        "denormalize_flag": args.denormalize,
        "batch_size": config["batch_size"],
    }

    np.savez_compressed(out_path, **save_dict)

    logger.info(f"Saved results to {out_path}")
    logger.info(f"  samples shape: {samples.shape}  (N, H, S)")
    logger.info(f"  ground_truth shape: {ground_truth.shape}  (N, H)")
    logger.info(f"  contexts shape: {contexts.shape}  (N, L)")
    logger.info(f"  full_trajectories shape: {test_trajectories[:N_windows].shape}  (N, T)")


if __name__ == "__main__":
    main()