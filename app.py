import argparse
import sys
from src.core.engine import ImageTransformerEngine
from src.ui.dashboard import create_dashboard

def main():
    parser = argparse.ArgumentParser(description="AI Image Transformer Production Launcher")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host ip to bind Gradio dashboard")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind Gradio dashboard")
    parser.add_argument("--share", action="store_true", help="Create public shareable link")
    args = parser.parse_args()

    print("Initializing Stable Diffusion XL + ControlNet Engine...")
    try:
        engine = ImageTransformerEngine()
        print("Model configurations loaded successfully.")
    except Exception as e:
        print(f"Error loading model engine config: {e}", file=sys.stderr)
        sys.exit(1)

    print("Initializing Gradio UI Dashboard...")
    dashboard = create_dashboard(engine)
    
    print(f"Launching production interface on http://{args.host}:{args.port}")
    dashboard.launch(server_name=args.host, server_port=args.port, share=args.share)

if __name__ == "__main__":
    main()
