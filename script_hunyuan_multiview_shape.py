import shutil
from gradio_client import Client, handle_file
c=Client("tencent/Hunyuan3D-2",token=__import__("os").environ["HF_TOKEN"])
r=c.predict(caption=None,image=handle_file("mv_front.png"),mv_image_front=handle_file("mv_front.png"),mv_image_back=handle_file("mv_back.png"),mv_image_left=handle_file("mv_left.png"),mv_image_right=handle_file("mv_right.png"),steps=50,guidance_scale=5,seed=1234,octree_resolution=200,check_box_rembg=False,num_chunks=8000,randomize_seed=False,api_name="/shape_generation")
shutil.copy(r[0]["value"],"mv_white.glb"); print(r[2])
