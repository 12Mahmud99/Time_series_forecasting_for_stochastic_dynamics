import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path
import logging
import yaml
import properscoring as ps
import datetime
from pathlib import Path
import time as timelib

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import pandas as pd
import argparse
import logging
import yaml
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import MetaData, TrainDatasets, FileDataset, ListDataset
from gluonts.evaluation import make_evaluation_predictions

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    get_next_file_num,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
import uncond_ts_diff.configs as diffusion_configs

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

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

def evaluate_guidance(
    config, model, test_dataset, transformation, num_samples=100
):
    logger.info(f"Evaluating with {num_samples} samples.")
    results = []
    if config["setup"] == "forecasting":
        missing_data_kwargs_list = [
            {
                "missing_scenario": "none",
                "missing_values": 0,
            }
        ]
        config["missing_data_configs"] = missing_data_kwargs_list
    elif config["setup"] == "missing_values":
        missing_data_kwargs_list = config["missing_data_configs"]
    else:
        raise ValueError(f"Unknown setup {config['setup']}")

    Guidance = guidance_map[config["sampler"]]
    sampler_kwargs = config["sampler_params"]
    for missing_data_kwargs in missing_data_kwargs_list:
        logger.info(
            f"Evaluating scenario '{missing_data_kwargs['missing_scenario']}' "
            f"with {missing_data_kwargs['missing_values']:.1f} missing_values."
        )
        sampler = Guidance(
            model=model,
            prediction_length=config["prediction_length"],
            num_samples=num_samples,
            **missing_data_kwargs,
            **sampler_kwargs,
        )


        transformed_testdata = transformation.apply(test_dataset, is_train=False)
        test_splitter = create_splitter(
            past_length=config["context_length"]+ max(model.lags_seq),
            future_length=config["prediction_length"],
            mode="test",
        )

        masking_transform = MaskInput(
            FieldName.TARGET,
            FieldName.OBSERVED_VALUES,
            config["context_length"],
            missing_data_kwargs["missing_scenario"],
            missing_data_kwargs["missing_values"],
        )
        test_transform = test_splitter + masking_transform
        
        print(f"BATCH SIZE {config['batch_size'],}")

        predictor = sampler.get_predictor(
            test_transform,
            #batch_size=1280 // num_samples,
            batch_size=config["batch_size"],
            device=config["device"],
        )
        
        
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=transformed_testdata,
            predictor=predictor,
            num_samples=num_samples,
        )
        
        forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
        
        tss = list(ts_it)

    return forecasts, tss

def read_config(
    config_path: Path,
    checkpoint_path: str,
    device: str,
) -> dict:
    """
    Loads a YAML training config and injects runtime parameters.

    Parameters
    ----------
    config_path     : path to yaml config file
    checkpoint_path : path to .ckpt file
    device          : torch device string e.g. "cuda" or "cpu"

    Returns
    -------
    config dict with ckpt and device injected
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config["ckpt"] is None:
        logger.warning("config[ckpt] reading from parameters ...") 
        config["ckpt"]   = checkpoint_path
    
    config["device"] = device

    logger.info(f"  dataset          : {config['dataset']}")
    logger.info(f"  prediction_length: {config['prediction_length']}")
    logger.info(f"  context_length   : {config['context_length']}")
    logger.info(f"  num_samples      : {config['num_samples']}")
    logger.info(f"  device           : {config['device']}")
    logger.info(f"  setup            : {config['setup']}")
    logger.info(f"  eval_every          : {config['eval_every']}")
    logger.info(f'batch_size  : {config["batch_size"]}')
    logger.info(f'ckpt  : {config["ckpt"]}')

    return config

def load_model(config: dict) -> TSDiff:
    """
    Initializes TSDiff architecture and loads checkpoint weights.
    """
    model = TSDiff(
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
    )

    #TEST3: loading the model's weights correctly
    logger.info("RUNNING TEST3")
    logger.info(f'Loading from {config["ckpt"]} ...')

    checkpoint = torch.load(config["ckpt"], map_location="cpu",weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
    model.load_state_dict(state_dict, strict=True)
    model = model.to(config["device"])
    model.eval()

    logger.info(f"Loaded checkpoint from: {config['ckpt']}")
    return model

import numpy as np
from gluonts.evaluation import make_evaluation_predictions

def predict_in_sample_chunks(dataset, predictor, total_num_samples, chunk_size, device):
    """
    Runs prediction in multiple passes of `chunk_size` samples each and
    concatenates along the sample axis, avoiding OOM from num_samples
    being multiplied directly into the per-step batch size.
    """
    all_forecasts_per_chunk = []
    n_chunks = (total_num_samples + chunk_size - 1) // chunk_size

    for c in range(n_chunks):
        this_chunk = min(chunk_size, total_num_samples - c * chunk_size)
        forecast_it, ts_it = make_evaluation_predictions(
            dataset=dataset,
            predictor=predictor,
            num_samples=this_chunk,
        )
        chunk_forecasts = list(forecast_it)
        all_forecasts_per_chunk.append(chunk_forecasts)

        if "cuda" in str(device):
            torch.cuda.empty_cache()  # release fragmented memory between chunks

    # ts_it is identical across chunks (same dataset), just take the last one
    n_windows = len(all_forecasts_per_chunk[0])
    merged_forecasts = []
    for w in range(n_windows):
        samples_concat = np.concatenate(
            [chunk[w].samples for chunk in all_forecasts_per_chunk], axis=0
        )  # (chunk_size * n_chunks, H) -> (total_num_samples, H)
        f = all_forecasts_per_chunk[0][w]
        f.samples = samples_concat  # reuse the Forecast object, replace samples
        merged_forecasts.append(f)

    return merged_forecasts, ts_it

def forecast(
    config      : dict,
    model       : TSDiff,
    windowed_dataset,
    transformation,
    time        : np.ndarray,
    train_test_split: int,
    full_trajectories: np.ndarray,  
) -> dict:
    """
    Runs TSDiff guidance inference and returns standardized forecast bundle.

    Returns
    -------
    dict with keys:
        samples           (N, T_pred, num_samples)
        ground_truth      (N, T_pred)
        full_trajectories (N, T_full)
        ci90_lower        (N, T_pred)
        ci90_upper        (N, T_pred)
        ci50_lower        (N, T_pred)
        ci50_upper        (N, T_pred)
        time_test         (T_pred,)
        time_train        (T_train,)
        train_test_split  int
        prediction_length int
        item_ids          (N,)
        time_elapsed      float (total)
        per_item_times    (N,)  per-batch-averaged time per trajectory
    """

    num_samples       = config["num_samples"]
    prediction_length = config["prediction_length"]
    context_length = config["context_length"]
    Guidance          = guidance_map[config["sampler"]]

    sampler = Guidance(
        model             = model,
        prediction_length = prediction_length,
        num_samples       = num_samples,
        missing_scenario  = "none",
        missing_values    = 0,
        **config["sampler_params"],
    )

    transformed_testdata = transformation.apply(windowed_dataset, is_train=False)

    test_splitter = create_splitter(
        past_length   = config["context_length"] + max(model.lags_seq),
        future_length = prediction_length,
        mode          = "test",
    )
    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        config["context_length"],
        "none",
        0,
    )
    test_transform = test_splitter + masking_transform

    predictor = sampler.get_predictor(
        test_transform,
        #batch_size = 1280 // num_samples,
        batch_size = config["batch_size"],
        device     = config["device"],
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset     = transformed_testdata,
        predictor   = predictor,
        num_samples = num_samples,
    )

    total_windows = len(list(transformed_testdata))

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

    # FIX: 'results' was referenced below but never defined — the timing loop
    # populates 'forecasts', not 'results'. Everything downstream needs this.
    results = forecasts

    time_elapsed = sum(per_item_times)
    logger.info(f"Inference took {time_elapsed:.2f}s total")
    logger.info(f"Per-batch avg time per trajectory: min={min(per_item_times):.3f}s, "
                f"max={max(per_item_times):.3f}s, mean={np.mean(per_item_times):.3f}s")

    forecast_samples = np.array([f.samples for f in results])
    print(f'forecast_samples : {forecast_samples.shape}')
    forecast_samples = np.transpose(forecast_samples, (0, 2, 1))
    print(f'forecast_samples : {forecast_samples.shape}')

    time_test  = time[train_test_split:train_test_split + prediction_length]
    time_train = time[:train_test_split]

    ci90_lower = np.percentile(forecast_samples, 5,  axis=2)   # (N, T_pred)
    ci90_upper = np.percentile(forecast_samples, 95, axis=2)
    ci50_lower = np.percentile(forecast_samples, 25, axis=2)
    ci50_upper = np.percentile(forecast_samples, 75, axis=2)

    # NOTE: removed the dead `plt.savefig(...)` call that was writing a blank
    # figure every run (the actual plotting block above it was commented out).
    # Restore the commented plotting block above if you want real diagnostic
    # plots saved here instead.

    item_ids = np.arange(len(results))

    ground_truth = full_trajectories[
        :len(results),
        train_test_split : train_test_split + prediction_length
    ]   # (N, H)

    # Context: exact context window
    contexts = full_trajectories[
        :len(results),
        train_test_split - context_length : train_test_split
    ]

    return {
        "full_trajectories" : full_trajectories[:len(results)],
        "samples"           : forecast_samples,   # (N, T_pred, S)
        "ground_truth"      : ground_truth,       # ONLY the forecast horizon
        "contexts"          : contexts,
        "ci90_lower"        : ci90_lower,         # (N, T_pred)
        "ci90_upper"        : ci90_upper,
        "ci50_lower"        : ci50_lower,
        "ci50_upper"        : ci50_upper,
        "time_test"         : time_test,          # (T_pred,)
        "time_train"        : time_train,         # (T_train,)
        "time"              : time,               # (T_full,)
        "train_test_split"  : train_test_split,   # forecast horizon start
        "prediction_length" : prediction_length,
        "context_length"    : context_length,
        "item_ids"          : item_ids,
        "time_elapsed"      : time_elapsed,             # total wall-clock time
        "per_item_times"    : np.array(per_item_times),  # per-batch-averaged time per trajectory
        "batch_size"        : config["batch_size"],
    }
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run TSDiff unconditional forecasting and save standardized bundle."
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to forecast config yaml (e.g. tsfdiff_forecast_config/double_well.yaml)",
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        help="Path to model checkpoint (e.g. lightning_logs/version_0/best_checkpoint.ckpt)",
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="Path to GluonTS dataset directory (e.g. gluonts_datasets/double_well)",
    )
    parser.add_argument(
        "--data",
        required=False,
        default=None,
        help="Path to original .npz data file from generator.py (for time array). Optional if time.npz exists in dataset_path.",
    )
    parser.add_argument(
        "--out", "-o",
        default="results/uncond_tsfdiff",
        help="Output directory or .npz path (e.g. results/uncond_tsfdiff)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available)",
    )
    args = parser.parse_args()
    
    #TEST1: correctly reading args 
    logger.info("RUNNING TEST1")
    logger.info(f"{args.config}")

    logger.info(f"Reading config: {args.config}")
    config = read_config(
        config_path     = Path(args.config),
        checkpoint_path = args.checkpoint,
        device          = args.device,
    )

    #TEST2:  corretly reading config
    logger.info("RUNNING TEST2")
    logger.info(f"{config['ckpt']}")
    logger.info(f"{config['prediction_length']}")
    torch.cuda.empty_cache()
    
    if not args.out:
        args.out=f"results/uncond_tsfdiff/{config['dataset']}"
        print(args.out)

    dataset_path = Path(args.dataset_path)
    with open(dataset_path / "metadata.json", "r") as f:
        meta_json = yaml.safe_load(f)

    freq = meta_json["freq"]

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if config.get("ckpt") is None:
        config["ckpt"] = args.checkpoint
    config["device"] = args.device

    L = config["context_length"]
    H = config["prediction_length"]
    metadata = MetaData(
        freq              = freq,
        prediction_length = H,  
    )
    test_ds  = FileDataset(dataset_path / "test",  freq=freq)
    train_ds = FileDataset(dataset_path / "train", freq=freq)
    dataset  = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

    test_trajectories = np.array([entry["target"] for entry in dataset.test])
    N, T = test_trajectories.shape
    logger.info(f"Full trajectories: {test_trajectories.shape}")

    time_npz = np.load(dataset_path / "time.npz")
    time = time_npz["time"]
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
    
    #TEST4: plot full trajectories and time
    '''plt.figure(figsize=(10,4))
    for i in range(8):
        plt.plot(train_time,full_trajectories[i,:train_test_split],color='blue')
        plt.plot(test_time,full_trajectories[i,train_test_split:],color='green')
    plt.savefig("TEST4: uncond_tsfdiff.png")'''

    

    #TEST4: ensure time is read  corretly
    '''logger.info("RUNNING TEST4")
    for key in time_npz.files:                 
        print(f"Key: {key}, Shape: {time_npz[key].shape}")
    logger.info("is time split correct?")
    reconstructed = np.concatenate([train_time, test_time])
    is_equal = np.array_equal(time, reconstructed)
    print(f"Are they equal? {is_equal}")'''

    # ---- 3. Load model --------------------------------------------------
    logger.info("Loading model...")
    model = load_model(config)
    

    # ---- 4. Setup transformation ----------------------------------------
    transformation = create_transforms(
        num_feat_dynamic_real = 0,
        num_feat_static_cat   = 0,
        num_feat_static_real  = 0,
        time_features         = model.time_features,
        prediction_length     = config["prediction_length"],
    )

    logger.info("Running forecast...")

    results = forecast(
        config           = config,
        model            = model,
        windowed_dataset     = windowed_ds,
        transformation   = transformation,
        time             = time,
        train_test_split = train_test_split,
        full_trajectories = test_trajectories,  
    )

    out_path = Path(args.out)

    # if out is a directory, auto-name the file using dataset name
    if out_path.suffix != ".npz":
        out_path.mkdir(parents=True, exist_ok=True)
        dataset_name = Path(args.dataset_path).name
        out_path = out_path / f"{dataset_name}.npz"
        logger.info(f"Output path: {out_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    

    np.savez_compressed(
        out_path,
        samples           = results["samples"], #(N, S, T)
        ground_truth      = results["ground_truth"], #(N, S, T)
        full_trajectories = results["full_trajectories"],
        ci90_lower        = results["ci90_lower"],
        ci90_upper        = results["ci90_upper"],
        ci50_lower        = results["ci50_lower"],
        ci50_upper        = results["ci50_upper"],
        time_test         = results["time_test"],
        time_train        = results["time_train"],
        time              = results["time"],
        train_test_split  = results["train_test_split"],
        prediction_length = results["prediction_length"],
        context_length = results["context_length"],
        item_ids          = results["item_ids"],
        time_elapsed      = results["time_elapsed"],
        batch_size = config["batch_size"],
        per_item_times   = results["per_item_times"], 
        num_samples       = config["num_samples"],
        
    )

    logger.info(f"Saved forecast bundle to: {out_path}")
    logger.info(f"  samples shape      : {results['samples'].shape}")
    logger.info(f"  ground_truth shape : {results['ground_truth'].shape}")

if __name__ == "__main__":
    main()