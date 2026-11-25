import shutil,sys
from gradio_client import Client, handle_file
c=Client("multimodalart/wan2-1-fast",token=__import__("os").environ["HF_TOKEN"],verbose=False)
r=c.predict(input_image=handle_file(sys.argv[1]),prompt="the man thinks and speaks, subtle natural head and hand movement, cinematic, smooth",height=640,width=384,negative_prompt="static, blurry, deformed, extra fingers, watermark, text",duration_seconds=3,guidance_scale=1.0,steps=6,seed=42,randomize_seed=False,api_name="/generate_video")
shutil.copy(r[0]["video"],sys.argv[2]); print("done")
