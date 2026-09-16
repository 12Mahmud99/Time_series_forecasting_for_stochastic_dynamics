"""
train/
======
Unified training module for the NFTSF comparison pipeline.

train.py         CLI entry point: ``python train/train.py --model X --config Y``
engine_pytorch.py  Custom PyTorch training loop (used by NFTSF).
engine_lightning.py  PyTorch Lightning dispatch (used by diffusion models).
"""
