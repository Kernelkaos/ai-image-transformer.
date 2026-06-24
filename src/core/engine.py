import os
import yaml
import torch
import numpy as np
import cv2
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    AutoencoderKL
)
from src.core.memory import VRAMGuard, flush_vram

class ImageTransformerEngine:
    def __init__(self, config_path="config/settings.yaml"):
        self.config = self._load_config(config_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.controlnet = None

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_models(self):
        """Loads and optimizes the SDXL and ControlNet pipelines."""
        if self.pipe is not None:
            return # Models already loaded
            
        settings = self.config["model_settings"]
        perf_settings = self.config["performance"]
        
        dtype = torch.float16 if settings["precision"] == "fp16" and self.device == "cuda" else torch.float32

        # 1. Load VAE to prevent black images issues in fp16
        vae = AutoencoderKL.from_pretrained(
            settings["vae_model"], 
            torch_dtype=dtype
        )

        # 2. Load ControlNet model
        self.controlnet = ControlNetModel.from_pretrained(
            settings["controlnet_canny_model"],
            torch_dtype=dtype
        )

        # 3. Load Main SDXL Pipeline
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            settings["sdxl_base_model"],
            controlnet=self.controlnet,
            vae=vae,
            torch_dtype=dtype
        )
        self.pipe.to(self.device)

        # 4. Apply performance optimizations
        if self.device == "cuda":
            if perf_settings.get("enable_xformers"):
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(f"Failed to enable xformers: {e}. Falling back to default attention.")
                    
            if perf_settings.get("enable_attention_slicing"):
                self.pipe.enable_attention_slicing()
                
            if perf_settings.get("enable_vae_tiling"):
                self.pipe.enable_vae_tiling()
                
            if perf_settings.get("enable_cpu_offload"):
                self.pipe.enable_sequential_cpu_offload()

    def process_canny_image(self, image: Image.Image, low_threshold=100, high_threshold=200):
        """Processes the input image to extract Canny edge lines."""
        # Convert PIL to OpenCV format (BGR)
        img_np = np.array(image.convert("RGB"))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        
        # Apply Canny
        canny_img = cv2.Canny(img_cv, low_threshold, high_threshold)
        
        # Convert back to PIL with 3 channels
        canny_img = canny_img[:, :, None]
        canny_img = np.concatenate([canny_img, canny_img, canny_img], axis=2)
        return Image.fromarray(canny_img)

    def generate(
        self, 
        input_image: Image.Image, 
        prompt: str, 
        negative_prompt: str = "",
        steps: int = None,
        guidance_scale: float = None,
        controlnet_scale: float = None,
        low_threshold: int = 100,
        high_threshold: int = 200,
        seed: int = -1
    ) -> Image.Image:
        """Runs the main SDXL + ControlNet Canny inference loop under VRAM guard."""
        self.load_models()
        
        constraints = self.config["inference_constraints"]
        steps = steps or constraints["default_steps"]
        guidance_scale = guidance_scale or constraints["default_guidance_scale"]
        controlnet_scale = controlnet_scale or constraints["controlnet_conditioning_scale"]

        # Ensure correct image sizing constraints
        w, h = input_image.size
        # Resize to fit constraints (typically 1024x1024 for SDXL)
        target_size = 1024
        if w != target_size or h != target_size:
            input_image = input_image.resize((target_size, target_size), Image.Resampling.LANCZOS)

        # 1. Process Canny Image
        canny_edges = self.process_canny_image(input_image, low_threshold, high_threshold)

        # 2. Setup Seed/Generator
        if seed != -1:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # 3. Run Inference under VRAM protection
        with VRAMGuard():
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=canny_edges,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=controlnet_scale,
                generator=generator
            )
            
        return output.images[0]
