import argparse
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pathlib import Path
import logging
from tqdm.auto import tqdm
import yaml
from concurrent.futures import ProcessPoolExecutor, as_completed
import time as timelib

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _fit_one_trajectory(
    i: int,
    train_array: np.ndarray,
    prediction_length: int,
    num_of_samples: int,
    context_length: int,
) -> tuple:
    """
    Fit one ARIMA model and return forecast samples for a single trajectory.
    Returns: (i, samples, ci90_lower, ci90_upper, ci50_lower, ci50_upper)
    """
    train_series = pd.Series(train_array)
    if context_length is not None:
        train_series = train_series.iloc[-context_length:]

    model = pm.auto_arima(train_series, metric="mse", random_state=42 + i)
    result = model.arima_res_

    sims = result.simulate(
        nsimulations=prediction_length,
        repetitions=num_of_samples,
        anchor="end",
        random_state=42 + i
    )
    sims = np.asarray(sims).squeeze()
    if sims.shape[0] == num_of_samples and sims.shape[1] == prediction_length:
        sims = sims.T

    ci90_lower = np.percentile(sims, 5, axis=1)
    ci90_upper = np.percentile(sims, 95, axis=1)
    ci50_lower = np.percentile(sims, 25, axis=1)
    ci50_upper = np.percentile(sims, 75, axis=1)

    return i, sims, ci90_lower, ci90_upper, ci50_lower, ci50_upper


def fit_and_forecast(
    positions: np.ndarray,
    train_test_split: int,
    num_of_samples: int,
    use_validation_set: bool,
    label: str,
    prediction_length: int = None,
    validation_metric: str = "mse",
    save_figures: bool = False,
    time: np.array = None,
    number_of_forecasts_to_plot: int = 6,
    path_to_save_figures: str = "./plots/ARIMA",
    context_length: int = None,
    n_jobs: int = 1,
) -> dict:
    """
    Fits one auto_arima model per trajectory, generates forecast samples
    and confidence intervals. Supports parallel execution with n_jobs > 1.
    """
    N = positions.shape[0]
    if prediction_length is None:
        prediction_length = positions.shape[1] - train_test_split

    full_trajectories = positions
    samples = np.zeros((N, prediction_length, num_of_samples))
    ground_truth = positions[:, train_test_split:train_test_split + prediction_length]
    contexts = full_trajectories[:,train_test_split - context_length : train_test_split]   # (N, L)
    
    ci90_lower_est = np.zeros((N, prediction_length))
    ci90_upper_est = np.zeros((N, prediction_length))
    ci50_lower_est = np.zeros((N, prediction_length))
    ci50_upper_est = np.zeros((N, prediction_length))

    train_arrays = [positions[i, :train_test_split].copy() for i in range(N)]

    time_start = timelib.time()

    if n_jobs == 1:
        for i in tqdm(range(N), desc="Fitting ARIMA models"):
            train_series = pd.Series(train_arrays[i])
            if context_length is not None:
                train_series = train_series.iloc[-context_length:]

            model = pm.auto_arima(train_series, metric=validation_metric, random_state=42 + i)
            result = model.arima_res_

            sims = result.simulate(
                nsimulations=prediction_length,
                repetitions=num_of_samples,
                anchor="end",
                random_state=42 + i
            )
            sims = np.asarray(sims).squeeze()
            if sims.shape[0] == num_of_samples and sims.shape[1] == prediction_length:
                sims = sims.T

            samples[i] = sims
            ci90_lower_est[i] = np.percentile(sims, 5, axis=1)
            ci90_upper_est[i] = np.percentile(sims, 95, axis=1)
            ci50_lower_est[i] = np.percentile(sims, 25, axis=1)
            ci50_upper_est[i] = np.percentile(sims, 75, axis=1)

    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for i in range(N):
                future = executor.submit(_fit_one_trajectory, i, train_arrays[i], prediction_length, num_of_samples,context_length)
                futures[future] = i

            for future in tqdm(as_completed(futures), total=N, desc="Fitting ARIMA models (parallel)"):
                i, sims, ci90_l, ci90_u, ci50_l, ci50_u = future.result()
                samples[i] = sims
                ci90_lower_est[i] = ci90_l
                ci90_upper_est[i] = ci90_u
                ci50_lower_est[i] = ci50_l
                ci50_upper_est[i] = ci50_u

    total_elapsed = timelib.time() - time_start
    logger.info(f"Total fitting time: {total_elapsed:.2f} seconds")

    if save_figures and time is not None:
        Path(path_to_save_figures).mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 4))
        for i in range(min(number_of_forecasts_to_plot, N)):
            plt.plot(time[:train_test_split + prediction_length],
                     full_trajectories[i, :train_test_split + prediction_length], color="blue")
        for i in range(min(number_of_forecasts_to_plot, N)):
            plt.plot(time[train_test_split:train_test_split + prediction_length],
                     full_trajectories[i, train_test_split:train_test_split + prediction_length], color="green")
        plt.xlabel("Time steps")
        plt.ylabel("Position")
        plt.title(f"ARIMA {label}")
        plt.legend()
        filename = datetime.datetime.now().strftime(f"full_trajectories_{label}_%Y%m%d_%H%M%S.png")
        plt.savefig(f"{path_to_save_figures}/{filename}")
        plt.close()

        fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(12, 12))
        axes = axes.flatten()
        for i in range(min(8, N)):
            axes[i].plot(time, full_trajectories[i])
            axes[i].plot(time[train_test_split:train_test_split + prediction_length],
                         ci50_upper_est[i], color='orange', alpha=0.5)
            axes[i].fill_between(time[train_test_split:train_test_split + prediction_length],
                                 ci90_lower_est[i], ci90_upper_est[i], color='orange', alpha=0.3)
            axes[i].axhline(y=1, color='red', linestyle='--', alpha=0.6)
            axes[i].axhline(y=-1, color='purple', linestyle='--', alpha=0.6)
            axes[i].set_xlabel("Time steps")
            axes[i].set_ylabel("Position")
        plt.tight_layout()
        detail_filename = datetime.datetime.now().strftime(f"ARIMA_forecasts_{label}_%Y%m%d_%H%M%S.png")
        plt.savefig(f"{path_to_save_figures}/{detail_filename}")
        plt.close()

    return {
        "samples": samples,
        "ground_truth": ground_truth,
        "ci90_lower": ci90_lower_est,
        "ci90_upper": ci90_upper_est,
        "ci50_lower": ci50_lower_est,
        "ci50_upper": ci50_upper_est,
        "full_trajectories": full_trajectories,
        "time_elapsed": total_elapsed,
        "contexts" : contexts
    }

def read_data(
    input_path: Path,
    split_override: int | None,
    predection_length_override: int | None,
) -> dict:
    """
    Loads a .npz data file produced by generator.py and returns a config dict.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix != ".npz":
        raise ValueError(f"Expected a .npz file, got: '{input_path.suffix}'")

    data = np.load(input_path, allow_pickle=False)

    positions = data["positions"]
    time = data["time"]
    train_test_split = int(data.get("train_test_split", 0))

    if split_override is not None:
        train_test_split = split_override
        logger.info(f"  train_test_split  : {train_test_split} (CLI override)")
    else:
        logger.info(f"  train_test_split  : {train_test_split} (from configs)")

    N, T = positions.shape

    return {
        "positions_test": positions,
        "time": time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fit one auto-ARIMA model per trajectory (parallel support)."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to .npz data file")
    parser.add_argument("--out", "-o", required=True, help="Output .npz path")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of Monte Carlo simulations")
    parser.add_argument("--split", type=int, default=None, help="Train/test split index (overrides .npz)")
    parser.add_argument("--prediction-length", type=int, default=None, help="Forecast horizon")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to forecasting config YAML")
    parser.add_argument("--num-models", "--models", type=int, default=3000, help="Number of ARIMA models to fit")
    parser.add_argument("--label", "-l", type=str, default="Unknown", help="Label for plots")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel processes (default 1)")
    parser.add_argument("--save-figures", action="store_true", help="Save plotting outputs")
    args = parser.parse_args()

    input_path = Path(args.input)
    logger.info(f"Reading input: {input_path}")
    input_data = read_data(input_path, args.split, args.prediction_length)

    positions = input_data["positions_test"]
    time = input_data["time"]


    use_validation_set = False
    context_length = None
    if args.config is not None:
        logger.info(f"Reading forecasting config from: {args.config}")
        with open(args.config, "r") as f:
            forecast_config = yaml.safe_load(f)
            prediction_length = forecast_config["prediction_length"]
            context_length = forecast_config["context_length"]
            use_validation_set = forecast_config["use_validation_set"]
            train_test_split = forecast_config["train_test_split"]

    N, T = positions.shape
    logger.info(f"Loaded data with {N} trajectories and {T} time steps.")
    logger.info(f"  num_samples = {args.num_samples}")
    logger.info(f"  prediction_length = {prediction_length}")
    logger.info(f"  train_test_split = {train_test_split}")
    logger.info(f"  n_jobs = {args.n_jobs}")

    if args.num_models is not None:
        positions = positions[:args.num_models]
        N = positions.shape[0]
        logger.info(f"  Limiting to first {N} trajectories")

    predection_length_override = args.prediction_length
    if predection_length_override is not None:
        prediction_length = predection_length_override
    logger.info(f"  prediction_length : {prediction_length}")
    logger.info(f"  context_length : {context_length}")


    results = fit_and_forecast(
        positions=positions,
        train_test_split=train_test_split,
        prediction_length=prediction_length,
        num_of_samples=args.num_samples,
        use_validation_set=use_validation_set,
        time=time,
        label=args.label,
        context_length=context_length,
        n_jobs=args.n_jobs,
        save_figures=args.save_figures,
        path_to_save_figures="./plots/ARIMA",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
        logger.warning(f"Changing output extension to .npz → {out_path}")

    np.savez(
        out_path,
        samples=results["samples"],
        ground_truth=results["ground_truth"],
        ci90_lower=results["ci90_lower"],
        ci90_upper=results["ci90_upper"],
        ci50_lower=results["ci50_lower"],
        ci50_upper=results["ci50_upper"],
        full_trajectories=results["full_trajectories"],
        time_test=time[train_test_split:train_test_split + prediction_length],
        time_train=time[:train_test_split],
        time=time,
        train_test_split=train_test_split,
        prediction_length=prediction_length,
        num_of_samples=args.num_samples,
        context_length = context_length,
        time_elapsed=results["time_elapsed"],
        contexts=results["contexts"]
    )

    logger.info(f"Saved forecast bundle to: {out_path}")
    logger.info(f"samples shape      : {results['samples'].shape}")
    logger.info(f"ground_truth shape : {results['ground_truth'].shape}")
    logger.info(f"total elapsed time : {results['time_elapsed']:.2f} s")


if __name__ == "__main__":
    main()