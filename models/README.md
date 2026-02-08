# Models

This directory contains training scripts for PSP-ML models.

## MNIST CNN

Train a simple CNN for MNIST digit classification:

```bash
# TODO: Add training script
# uv run train_mnist.py
```

The trained model should be exported to TFLite format and placed in `examples/mnist-bench/`.

## Testing Inference

Use `run_inference.py` to verify the TFLite model works correctly before deploying to PSP:

```bash
uv run run_inference.py
```
