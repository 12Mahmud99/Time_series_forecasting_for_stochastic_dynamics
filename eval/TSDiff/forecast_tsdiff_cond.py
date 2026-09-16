import argparse
import logging
import yaml
import numpy as np
import torch
import datetime
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import time as timelib

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import MetaData, TrainDatasets, FileDataset, ListDataset
from gluonts.evaluation import make_evaluation_predictions

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)
from uncond_ts_diff.model import TSDiffCond
import uncond_ts_diff.configs as diffusion_configs
import pandas as pd

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(config: dict) -> TSDiffCond:
    model = TSDiffCond(
        **getattr(
            diffusion_configs,
            config.get("diffusion_config", "diffusion_small_config"),
        ),
        freq              = config["freq"],
        use_features      = config["use_features"],
        use_lags          = config["use_lags"],
        context_length    = config["context_length"],
        prediction_length = config["prediction_length"],
        init_skip         = config["init_skip"],
        noise_observed    = config.get("noise_observed", True),
    )

    logger.info(f"Loading from {config['ckpt']} ...")
    checkpoint = torch.load(config["ckpt"], map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
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

    This forces GluonTS test-mode splitter to pick exactly:
        context  = [train_test_split - context_length : train_test_split]
        forecast = [train_test_split                  : train_test_split + prediction_length]

    Why this works
    --------------
    GluonTS create_splitter in "test" mode uses the LAST
    (past_length + future_length) steps of each series.
    By truncating each series to exactly that window, we guarantee
    the splitter always picks the window we want — regardless of what
    prediction_length was used when the FileDataset was originally created.

    No files are written; this is purely in-memory.

    Parameters
    ----------
    full_trajectories : (N, T)  full test trajectories
    train_test_split  : int     step index where forecast starts (e.g. 900)
    context_length    : int     L (e.g. 25)
    prediction_length : int     H (e.g. 25)
    freq              : str     GluonTS frequency string (e.g. "H")

    Returns
    -------
    ListDataset with N entries, each of length context_length + prediction_length
    """
    start_idx = train_test_split - context_length   # e.g. 875 for 25/25
    end_idx   = train_test_split + prediction_length # e.g. 925 for 25/25

    assert start_idx >= 0, \
        f"train_test_split ({train_test_split}) < context_length ({context_length})"
    assert end_idx <= full_trajectories.shape[1], \
        (f"train_test_split ({train_test_split}) + prediction_length ({prediction_length})"
         f" = {end_idx} > T ({full_trajectories.shape[1]})")

    entries = [
        {
            FieldName.TARGET:  traj[start_idx:end_idx].astype(np.float32),
            FieldName.START:   pd.Timestamp("2000-01-01"), 
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


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def forecast(
    config            : dict,
    model             : TSDiffCond,
    windowed_dataset,      
    transformation,
    time              : np.ndarray,
    train_test_split  : int,
    full_trajectories : np.ndarray,
) -> dict:
    """
    Run TSDiffCond inference on the windowed dataset.

    Because each series in windowed_dataset has length exactly
    context_length + prediction_length, the test splitter will always
    pick the correct window without any ambiguity.
    """
    num_samples       = config["num_samples"]
    prediction_length = config["prediction_length"]
    context_length    = config["context_length"]

    transformed_testdata = transformation.apply(windowed_dataset, is_train=False)

    test_splitter = create_splitter(
        past_length   = context_length + max(model.lags_seq),
        future_length = prediction_length,
        mode          = "test",
    )

    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        context_length,
        "none",
        0,
    )
    test_transform = test_splitter + masking_transform

    predictor = model.get_predictor(
        test_transform,
        batch_size = 1280,
        device     = config["device"],
    )

    
    
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset     = transformed_testdata,
        predictor   = predictor,
        num_samples = num_samples,
    )
    
    time_start = timelib.time()
    results = list(tqdm(forecast_it, total=len(list(windowed_dataset))))
    
    time_elapsed = timelib.time() - time_start
    print(f'time_elapsed : {time_elapsed}')

    
    logger.info(f"samples per forecast: {results[0].samples.shape}")

    forecast_samples = np.array([f.samples for f in results])
    forecast_samples = np.transpose(forecast_samples, (0, 2, 1))  # (N, H, S)

    # Time arrays for the actual window used
    time_test  = time[train_test_split : train_test_split + prediction_length]
    time_train = time[:train_test_split]

    ci90_lower = np.percentile(forecast_samples, 5,  axis=2)   # (N, H)
    ci90_upper = np.percentile(forecast_samples, 95, axis=2)
    ci50_lower = np.percentile(forecast_samples, 25, axis=2)
    ci50_upper = np.percentile(forecast_samples, 75, axis=2)

    # Ground truth: exact forecast window from full trajectories
    ground_truth = full_trajectories[
        :len(results),
        train_test_split : train_test_split + prediction_length
    ]   # (N, H)

    # Context: exact context window
    contexts = full_trajectories[
        :len(results),
        train_test_split - context_length : train_test_split
    ]   # (N, L)

    return {
        "samples"           : forecast_samples,          # (N, H, S)
        "ground_truth"      : ground_truth,              # (N, H)
        "contexts"          : contexts,                  # (N, L)
        "full_trajectories" : full_trajectories[:len(results)],  # (N, T)
        "ci90_lower"        : ci90_lower,
        "ci90_upper"        : ci90_upper,
        "ci50_lower"        : ci50_lower,
        "ci50_upper"        : ci50_upper,
        "time_test"         : time_test,
        "time_train"        : time_train,
        "time"              : time,
        "train_test_split"  : train_test_split,
        "prediction_length" : prediction_length,
        "context_length"    : context_length,
        "item_ids"          : np.arange(len(results)),
        "time_elapsed" : time_elapsed
    }


def main():
    parser = argparse.ArgumentParser(
        description="TSDiff-Cond forecast — flexible L/H without GluonTS regeneration."
    )
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--dataset_path", required=True,
                        help="Path to GluonTS dataset directory (used for freq and time.npz)")
    parser.add_argument("--out", "-o", default="results/cond_tsfdiff")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train_test_split" , "-tts", type=int,default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if config.get("ckpt") is None:
        config["ckpt"] = args.checkpoint
    config["device"] = args.device

    L = config["context_length"]
    H = config["prediction_length"]

    logger.info(f"dataset          : {config['dataset']}")
    logger.info(f"context_length   : {L}")
    logger.info(f"prediction_length: {H}")
    logger.info(f"ckpt             : {config['ckpt']}")

    dataset_path = Path(args.dataset_path)
    with open(dataset_path / "metadata.json", "r") as f:
        meta_json = yaml.safe_load(f)

    freq = meta_json["freq"]

    # Load full trajectories from the FileDataset (full 1000-step series)
    # We use prediction_length from CONFIG not metadata so GluonTS doesn't
    # interfere with the window size.
    metadata = MetaData(
        freq              = freq,
        prediction_length = H,   
    )
    test_ds  = FileDataset(dataset_path / "test",  freq=freq)
    train_ds = FileDataset(dataset_path / "train", freq=freq)
    dataset  = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

    # Full trajectories — shape (N, T), T=1000
    test_trajectories = np.array([entry["target"] for entry in dataset.test])
    N, T = test_trajectories.shape
    logger.info(f"Full trajectories: {test_trajectories.shape}")

    # Time array and train_test_split
    time_npz         = np.load(dataset_path / "time.npz")
    time             = time_npz["time"]
    
    if args.train_test_split is not None:
        train_test_split = int(args.train_test_split)    
        print(f'using custom train test split {train_test_split}')
    else:
        train_test_split = int(time_npz["train_test_split"])
    
    logger.info(f"train_test_split : {train_test_split}")
    logger.info(f"Context window   : steps {train_test_split-L}–{train_test_split-1}")
    logger.info(f"Forecast window  : steps {train_test_split}–{train_test_split+H-1}")

    # Verify the requested window fits within the data
    assert train_test_split - L >= 0, \
        f"Context window starts at {train_test_split-L} which is before the start of data"
    assert train_test_split + H <= T, \
        f"Forecast window ends at {train_test_split+H} which exceeds T={T}"

    windowed_ds = make_windowed_dataset(
        full_trajectories = test_trajectories,
        train_test_split  = train_test_split,
        context_length    = L,
        prediction_length = H,
        freq              = freq,
    )

    model = load_model(config)

    transformation = create_transforms(
        num_feat_dynamic_real = 0,
        num_feat_static_cat   = 0,
        num_feat_static_real  = 0,
        time_features         = model.time_features,
        prediction_length     = H,
    )

    logger.info("Running forecast...")
    
    results = forecast(
        config= config,
        model = model,
        windowed_dataset  = windowed_ds,
        transformation = transformation,
        time = time,
        train_test_split  = train_test_split,
        full_trajectories = test_trajectories,
    )
    


    logger.info("Verifying ground truth alignment...")
    gt_check = results["ground_truth"]
    ft_check = test_trajectories[:len(gt_check),
                                  train_test_split:train_test_split + H]
    assert np.allclose(gt_check, ft_check, atol=1e-4), \
        "Ground truth mismatch — check train_test_split and prediction_length"
    logger.info("Ground truth alignment verified ✓")

    ctx_check = results["contexts"]
    ctx_expected = test_trajectories[:len(ctx_check),
                                      train_test_split-L:train_test_split]
    assert np.allclose(ctx_check, ctx_expected, atol=1e-4), \
        "Context mismatch — check train_test_split and context_length"
    logger.info("Context alignment verified ✓")

    out_path = Path(args.out)
    if out_path.suffix != ".npz":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"{config['dataset']}.npz"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        samples = results["samples"],           # (N, H, S)
        ground_truth = results["ground_truth"],      # (N, H)
        contexts = results["contexts"],           # (N, L)
        full_trajectories = results["full_trajectories"], # (N, T)
        ci90_lower = results["ci90_lower"],
        ci90_upper = results["ci90_upper"],
        ci50_lower = results["ci50_lower"],
        ci50_upper = results["ci50_upper"],
        time_test = results["time_test"],
        time_train = results["time_train"],
        time = results["time"],
        train_test_split = results["train_test_split"],
        prediction_length = results["prediction_length"],
        context_length = results["context_length"],
        item_ids = results["item_ids"],
        time_elapsed =  results["time_elapsed"]
    )

    logger.info(f"Saved: {out_path}")
    logger.info(f"  samples      : {results['samples'].shape}  (N, H, S)")
    logger.info(f"  ground_truth : {results['ground_truth'].shape}  (N, H)")
    logger.info(f"  contexts     : {results['contexts'].shape}  (N, L)")


if __name__ == "__main__":
    main()