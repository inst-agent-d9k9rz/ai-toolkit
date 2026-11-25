import shutil,sys
from gradio_client import Client, handle_file
c=Client("Lightricks/ltx-video-distilled",token=__import__("os").environ["HF_TOKEN"],verbose=False)
r=c.predict(prompt="the man thinks and speaks, subtle natural head and hand movement, cinematic",negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",input_image_filepath=handle_file(sys.argv[1]),input_video_filepath=None,height_ui=704,width_ui=416,mode="image-to-video",duration_ui=3,ui_frames_to_use=9,seed_ui=42,randomize_seed=False,ui_guidance_scale=1,improve_texture_flag=True,api_name="/image_to_video")
shutil.copy(r[0]["video"],sys.argv[2]); print("done")
