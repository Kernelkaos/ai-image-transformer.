import gradio as gr
from PIL import Image
from src.core.engine import ImageTransformerEngine
from src.core.memory import get_vram_info

def create_dashboard(engine: ImageTransformerEngine):
    def run_inference(
        image, 
        prompt, 
        neg_prompt, 
        steps, 
        guidance, 
        control_scale,
        low_thresh,
        high_thresh,
        seed
    ):
        if image is None:
            raise gr.Error("Please upload an input image first.")
        if not prompt:
            raise gr.Error("Please provide a prompt description.")
            
        # Run generation
        output = engine.generate(
            input_image=image,
            prompt=prompt,
            negative_prompt=neg_prompt,
            steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_scale=float(control_scale),
            low_threshold=int(low_thresh),
            high_threshold=int(high_thresh),
            seed=int(seed)
        )
        
        # Get VRAM usage
        vram = get_vram_info()
        vram_status = (
            f"Device: {vram['device']} | "
            f"Allocated: {vram.get('allocated_mb', 0)} MB | "
            f"Cached: {vram.get('cached_mb', 0)} MB"
        )
        
        return output, vram_status

    # Define Theme and Styling
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate"
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_primary_text_color="white"
    )

    with gr.Blocks(theme=theme, title="AI Image Transformer Dashboard") as interface:
        gr.Markdown(
            """
            # 🎨 AI Image Transformer (SDXL + ControlNet Canny)
            Generate production-grade image variations keeping structure constraints.
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Source Image", type="pil")
                prompt = gr.Textbox(
                    label="Prompt", 
                    placeholder="Describe the target image styling (e.g., 'a cinematic cyberpunk street, high detail, 8k')..."
                )
                neg_prompt = gr.Textbox(
                    label="Negative Prompt", 
                    placeholder="Things to avoid (e.g., 'blurry, low quality, deformed, extra limbs')..."
                )
                
                with gr.Accordion("Advanced Parameters", open=False):
                    steps = gr.Slider(minimum=10, maximum=100, step=1, value=30, label="Inference Steps")
                    guidance = gr.Slider(minimum=1.0, maximum=20.0, step=0.5, value=7.5, label="Guidance Scale")
                    control_scale = gr.Slider(
                        minimum=0.0, maximum=2.0, step=0.05, value=0.5, 
                        label="ControlNet Canny Influence"
                    )
                    low_thresh = gr.Slider(minimum=1, maximum=255, step=1, value=100, label="Canny Low Threshold")
                    high_thresh = gr.Slider(minimum=1, maximum=255, step=1, value=200, label="Canny High Threshold")
                    seed = gr.Number(value=-1, precision=0, label="Seed (-1 for random)")
                
                generate_btn = gr.Button("Transform Image", variant="primary")
                
            with gr.Column(scale=1):
                output_image = gr.Image(label="Transformed Image", type="pil")
                vram_display = gr.Textbox(
                    label="Hardware Status", 
                    value="Inference idle. Device info will update on generation.", 
                    interactive=False
                )
                
        # Event mapping
        generate_btn.click(
            fn=run_inference,
            inputs=[
                input_image, 
                prompt, 
                neg_prompt, 
                steps, 
                guidance, 
                control_scale,
                low_thresh,
                high_thresh,
                seed
            ],
            outputs=[output_image, vram_display]
        )
        
    return interface
