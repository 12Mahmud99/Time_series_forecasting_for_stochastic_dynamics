# train_csdi.py
import argparse
import json
import os
import sys
import yaml
import torch
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))                   
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))        

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "CSDI"))   
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))           
sys.path.insert(0, PROJECT_ROOT)                                          

from CSDI.main_model import CSDI_Forecasting
from CSDI.dataset_md import get_dataloader_md
from CSDI.utils import train

train = train

def main():
    parser = argparse.ArgumentParser(description="Train CSDI on MD trajectories")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--out", "-o", required=True)
    parser.add_argument("--device", "-d", default="cuda:0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["model"]["is_unconditional"] = 0
    stride = config["train"]["stride"]
    context_length = config["train"]["context_length"]
    prediction_length = config["train"]["prediction_length"]
    batch_size = config["train"]["batch_size"]
    val_size = config["train"]["val_size"]
    test_size = config["train"]["test_size"]
    valid_epoch_interval=config["train"]["valid_epoch_interval"]
    z=config["train"]["training_curv"]
    
    
    print(json.dumps(config, indent=4))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    train_loader, val_loader = get_dataloader_md(
        flag="train",
        npz_path = args.input,
        context_length = context_length,
        prediction_length = prediction_length,
        batch_size = batch_size,
        val_size = val_size,
        test_size = test_size,
        stride = stride
    )
    '''print(f"std={std:.4f}  train={len(train_loader)}  "
          f"val={len(val_loader)}  test={len(test_loader)}")'''

    #np.save(os.path.join(args.out, "std.npy"), np.array([std]))
    np.save(os.path.join(args.out, "context_length.npy"), np.array([context_length]))
    np.save(os.path.join(args.out, "prediction_length.npy"), np.array([prediction_length]))

    model = CSDI_Forecasting(config, args.device, target_dim=1).to(args.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, config["train"], train_loader, valid_loader = val_loader, valid_epoch_interval = valid_epoch_interval, foldername = args.out)
    print(f"Saved to: {args.out}/model.pth")

if __name__ == "__main__":
    main()