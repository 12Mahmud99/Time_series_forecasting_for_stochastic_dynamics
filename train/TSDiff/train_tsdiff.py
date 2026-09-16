# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

import pykeops
import yaml
import torch
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, EarlyStopping

from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName

import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.model.callback import EvaluateCallback
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)

from gluonts.dataset.split import split
from gluonts.dataset.common import (
    MetaData,
    TrainDatasets,
    FileDataset,
)

#for loading custom data sets
from gluonts.dataset.common import load_datasets
from pathlib import Path

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}


def create_model(config):
    model = TSDiff(
        **getattr(diffusion_configs, config["diffusion_config"]),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization=config["normalization"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        lr=config["lr"],
        init_skip=config["init_skip"],
    )
    
    model.to(config["device"])
    return model

def main(config, log_dir, dataset_path,running_tests):
    # Load parameters
    dataset_name = config["dataset"]
    freq = config["freq"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]
    last_window = context_length + prediction_length
    #T = config["total_time_steps"]
    #T = 1000

    # Create model
    model = create_model(config)

    with open(dataset_path / "metadata.json", "r") as f:
        meta_json = yaml.safe_load(f)
    metadata = MetaData(freq=meta_json["freq"], prediction_length=meta_json["prediction_length"])
    train_ds = FileDataset(dataset_path / "train", freq=metadata.freq)
    test_ds = FileDataset(dataset_path / "test", freq=metadata.freq)
    dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)
    #print(f"dataset.metadata.freq: {dataset.metadata.freq}")
    #print(f"dataset.metadata.prediction_length: {dataset.metadata.prediction_length}")
    #print(f'train_set length: {len(dataset.train)}')
    #print(f'test_set length: {len(dataset.test)}')

    #ensure they're equal
    #print(dataset.metadata.prediction_length)
    #print(prediction_length)

    #assert dataset.metadata.freq == freq
    #assert dataset.metadata.prediction_length == prediction_length

    if config["setup"] == "forecasting":
        training_data = dataset.train
    elif config["setup"] == "missing_values":
        missing_values_splitter = OffsetSplitter(offset=-last_window)
        training_data, _ = missing_values_splitter.split(dataset.train)

    if running_tests:
        logger.info("RUNNING TEST #1: Confimr that the data read/written is correct")

        iterator = iter(dataset.test)
        testing_size=len(dataset.test)
        for i in range(testing_size):
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
        logger.info("COMPLETED TEST1")

    num_rolling_evals = int(len(dataset.test) / len(dataset.train))
    if num_rolling_evals == 0:
        num_rolling_evals=1 ##prevent 0 validation data set
    logger.info(f"----------------------------num_rolling_evals : {num_rolling_evals}")
    
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
        mode="train", ##if training it uses ExpectedNumInstanceSampler which samples window randomly
    )
    
    callbacks = []
    if config["use_validation_set"]:
        
        #train_val_splitter = OffsetSplitter(offset=-int(0.1*T) * num_rolling_evals)
        train_val_splitter = OffsetSplitter(offset=-config["prediction_length"]*num_rolling_evals)
        train_data_post, val_gen = train_val_splitter.split(training_data)
        
        print(f'train_val_splitter: {type(train_val_splitter)}')
        print(f'val_gen: {type(val_gen)}')
        val_data = val_gen.generate_instances(
            config["prediction_length"], 
            windows=num_rolling_evals #sliding window for validation only
        )
        
        ##only training data post split is used for training, validation data is only used for evaluation in the callback, never for training
        transformed_data = transformation.apply(train_data_post, is_train=True)
        
        if running_tests:
            logger.info("RUNNING TEST #2: Confirm Train/Validation Split")       
            first_original = next(iter(training_data))["target"]
            first_train = next(iter(train_data_post))["target"]
            
            #print(f"Match (should be True): {first_train[-1] == first_original[118]}")
            #print(f"No overlap (should be True): {first_train[-1] != first_original[119]}")

            #plotting 
            logger.info("RUNNING TEST #2: Confirm Train/Validation Split -- Plotting")  
            plt.plot(read_time[:read_split],gluonts_x_np[:read_split], color='blue')
            plt.plot(read_time[read_split:],gluonts_x_np[read_split:], color='green')
        
        callbacks = [
            EvaluateCallback(
                context_length=config["context_length"],
                prediction_length=config["prediction_length"],
                sampler=config["sampler"],
                sampler_kwargs=config["sampler_params"],
                num_samples=config["num_samples"],
                model=model,
                transformation=transformation,
                test_dataset=dataset.test,
                val_dataset=val_data, ## dataset
                eval_every=config["eval_every"],
            )
        ]
    else:
        transformed_data = transformation.apply(training_data, is_train=True)

    print(transformed_data)
    
    #raise SystemExit("ENDED------------------------------------------------------------------")

    #log_monitor = "train_loss"
    log_monitor="valid_loss"
    filename = dataset_name + "-{epoch:03d}-{train_loss:.3f}"

    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor=f"{log_monitor}",
        mode="min",
        filename=filename,
        save_last=True,
        save_weights_only=True,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",   
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
    print("-----------------------------------------------------------------------------------")
    logger.info(f"Logging to {trainer.logger.log_dir}")
    trainer.fit(model, train_dataloaders=data_loader)
    
    logger.info("Training completed.")

if __name__ == "__main__":
    # Setup Logger
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
    
    parser.add_argument(
        "--run_tests", "-t",
        default=False,
        type=bool,
        help=("Run tests for debugging")
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
    running_tests=bool(args.run_tests)
    main(config=config, log_dir=args.out_dir, dataset_path=dataset_path, running_tests=args.run_tests)