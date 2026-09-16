# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
from pathlib import Path

import yaml
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping

from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.model import TSDiffCond
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
    ConcatDataset,
#    linear_beta_schedule
)

from gluonts.dataset.common import (
    MetaData,
    TrainDatasets,
    FileDataset,
)

import numpy as np
import matplotlib.pyplot as plt

#with torch.serialization.safe_globals([linear_beta_schedule]):
#    checkpoint = torch.load("your_model.pt", weights_only=True)

def create_model(config):
    model = TSDiffCond(
        **getattr(diffusion_configs, config["diffusion_config"]),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization=config["normalization"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        lr=config["lr"],
        init_skip=config["init_skip"],
        noise_observed=config["noise_observed"],
    )
    model.to(config["device"])
    return model


def evaluate_conditional(
    config,
    model: TSDiffCond,
    test_dataset,
    transformation,
    num_samples=1000,
):
    logger.info(f"Evaluating with {num_samples} samples.")
    logger.info(
        f"Evaluating scenario '{config['missing_scenario']}' "
        f"with {config['missing_values']:.1f} missing_values."
    )

    results = []

    transformed_testdata = transformation.apply(test_dataset, is_train=False)
    test_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="test",
    )

    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        config["context_length"],
        config["missing_scenario"],
        config["missing_values"],
    )
    test_transform = test_splitter + masking_transform

    predictor = model.get_predictor(
        test_transform,
        batch_size=1280,
        device=config["device"],
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata,
        predictor=predictor,
        num_samples=num_samples,
    )
    forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
    tss = list(ts_it)
    evaluator = Evaluator()
    metrics, _ = evaluator(tss, forecasts)
    metrics = filter_metrics(metrics)
    results.append(dict(**metrics))

    return results


def main(config, log_dir,dataset_path):
    # Load parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
#    train_test_split = config["train_test_split"]
    total_length = context_length + prediction_length
    #T=config.get("total_time_steps",1000)
    #T=1000

    # Create model
    model = create_model(config)

    with open(dataset_path / "metadata.json", "r") as f:
        meta_json = yaml.safe_load(f)
    metadata = MetaData(freq=meta_json["freq"], prediction_length=meta_json["prediction_length"])
    train_ds = FileDataset(dataset_path / "train", freq=metadata.freq)
    test_ds = FileDataset(dataset_path / "test", freq=metadata.freq)
    dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

    #ensure they're equal
    #print(dataset.metadata.prediction_length)
    #print(prediction_length)

    #assert dataset.metadata.freq == freq
    #assert dataset.metadata.prediction_length == prediction_length

    if config["setup"] == "forecasting":
        training_data = dataset.train
    elif config["setup"] == "missing_values":
        missing_values_splitter = OffsetSplitter(offset=-total_length)
        training_data, _ = missing_values_splitter.split(dataset.train)

    #TEST1
    
    #To ensure that it is correctly reading the data
    logger.info("RUNNING TEST1")

    iterator = iter(dataset.test)
    
    '''for i in range(6):
        gluonts_x_map=next(iterator)
                                   #index=gluonts_x_map["item_id"])
        gluonts_t_dict=np.load(dataset_path/"time.npz")
        read_time=gluonts_t_dict["time"] ##stores as np.array
        read_split=int(gluonts_t_dict["train_test_split"])
        gluonts_x_np = np.array(gluonts_x_map["target"])
        plt.plot(read_time[:read_split],gluonts_x_np[:read_split], color='blue')
        plt.plot(read_time[read_split:],gluonts_x_np[read_split:], color='green')
    
    
    plt.title(".npz data")
    plt.savefig("TEST1: train_model_custom.png")
    logger.info("COMPLETED TEST1")'''

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))
    if num_rolling_evals==0:
        num_rolling_evals = 1

    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=model.time_features,
        prediction_length=config["prediction_length"],
    )

    training_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="train",
    )

    if config["setup"] == "forecasting":
        config["missing_scenario"] = "none"
        config["missing_values"] = 0

    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        config["context_length"],
        config.get("train_missing_scenario", config["missing_scenario"]),
        config["missing_values"],
    )
    train_transform = training_splitter + masking_transform

    callbacks = []
    val_loader = None
    if config["use_validation_set"]:
        #print(f"cardinality: {dataset.metadata}")
        
        train_val_splitter = OffsetSplitter(
            offset=-config["prediction_length"] * num_rolling_evals # for traffic and electricity
            #offset=-int(T * val_por) * num_rolling_evals
            #offset=-config["prediction_length"] #for synthetic data and md
        )
        train_data_post, val_gen = train_val_splitter.split(training_data)
        transformed_data = transformation.apply(train_data_post, is_train=True)

        val_dataset = ConcatDataset(
            val_gen.generate_instances(
                config["prediction_length"], num_rolling_evals
            )
        )
        val_splitter = create_splitter(
            past_length=config["context_length"] + max(model.lags_seq),
            future_length=config["prediction_length"],
            mode="val",
        )
        transformed_valdata = transformation.apply(val_dataset, is_train=False)
        val_loader = ValidationDataLoader(
            transformed_valdata,
            batch_size=1280,
            stack_fn=batchify,
            transform=val_splitter + masking_transform,
        )

        callbacks = []
        log_monitor = "valid_loss"
    else:
        transformed_data = transformation.apply(training_data, is_train=True)
        log_monitor = "train_loss"

    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"

    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=train_transform,
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
       # save_weights_only=False,
       )
    
    early_stop_callback = EarlyStopping(
        monitor="valid_loss",   
        patience=10,            
        mode="min",             
        verbose=True            
    )
    callbacks.append(early_stop_callback)
    callbacks.append(checkpoint_callback)
    #callbacks.append(RichProgressBar())
    
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else None,
        devices=[int(config["device"].split(":")[-1])],
        max_epochs=config["max_epochs"],
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        callbacks=callbacks,
        default_root_dir=log_dir,
        gradient_clip_val=config.get("gradient_clip_val", None),
        check_val_every_n_epoch=config["eval_every"],
    )
    logger.info(f"Logging to {trainer.logger.log_dir}")
    trainer.fit(
        model, train_dataloaders=data_loader, val_dataloaders=val_loader
    )
    logger.info("Training completed.")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--dataset_path", type=str, default="./", help="Path to dataset dir"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./", help="Path to results dir"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    dataset_path=Path(args.dataset_path)
    main(config=config, log_dir=args.out_dir, dataset_path=dataset_path)
