"""Registry of the AI models/services this toolkit can drive.
Token: set HF_TOKEN in the environment (never commit it)."""
import os
from gradio_client import Client, handle_file

REGISTRY = {
    "image_to_3d_shape": {"space": "tencent/Hunyuan3D-2", "api": "/shape_generation", "note": "untextured GLB; texture step broken on public Space"},
    "image_to_video_ltx": {"space": "Lightricks/ltx-video-distilled", "api": "/image_to_video", "note": "about 3 s clips, no audio, free ZeroGPU quota"},
    "image_to_video_wan": {"space": "multimodalart/wan2-1-fast", "api": "/generate_video", "note": "failed with RuntimeError on first try"},
    "web_pixverse": {"site": "https://app.pixverse.ai", "note": "free: 60 start credits + 30/day; web only"},
    "web_google_flow": {"site": "https://flow.google.com", "note": "free: 50 credits/day, needs user's own Google account"},
}

def client(name):
    spec = REGISTRY[name]
    return Client(spec["space"], token=os.environ.get("HF_TOKEN"), verbose=False)
