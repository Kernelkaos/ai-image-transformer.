# AI Image Transformer

A production-grade image translation pipeline utilizing **Stable Diffusion XL (SDXL) Base 1.0** and **ControlNet Canny 1.0** to generate high-fidelity image variations while preserving structural edge constraints.

## Architecture

This project is built using a decoupled, modular design pattern:
- `config/`: Declares settings, hyperparameters, and device performance limits.
- `src/core/memory.py`: Dedicated garbage collection and PyTorch CUDA VRAM optimization guard layers.
- `src/core/engine.py`: Image loading, OpenCV-based Canny processing, and SDXL+ControlNet inference orchestration.
- `src/ui/dashboard.py`: Gradio web dashboard mapping input parameters and visualizing real-time hardware allocations.
- `app.py`: Standard entry point CLI launcher.

## Requirements

Ensure CUDA is correctly configured. CPU execution is supported but highly discouraged due to SDXL latency constraints.

```bash
pip install -r requirements.txt
```

## Running the Application

Start the web dashboard locally:

```bash
python3 app.py
```

Arguments:
- `--host`: Address to bind server (default: `127.0.0.1`)
- `--port`: Port to listen (default: `7860`)
- `--share`: Generate public Gradio sharing URL

## Optimizations Enabled

- **FP16 Mixed Precision**: Halves VRAM usage.
- **xFormers Attention**: Memory-efficient attention processing.
- **Attention Slicing & VAE Tiling**: Enables high-resolution inference on consumer GPUs.
- **Active Garbage Collection**: Proactively purges unused PyTorch memory nodes before and after inference.
