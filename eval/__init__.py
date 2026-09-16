"""
eval/
=====
Shared evaluation module for the NFTSF comparison pipeline.

evaluate.py loads any trained model checkpoint, runs inference on the
held-out test trajectories, computes canonical metrics (MAE, RMSE, CRPS,
CI90 / CI50 coverage), and writes a ``results.npz`` in the canonical schema.
"""
