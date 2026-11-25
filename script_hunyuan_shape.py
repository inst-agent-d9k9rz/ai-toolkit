from gradio_client import Client, handle_file
c=Client("tencent/Hunyuan3D-2", token=__import__("os").environ["HF_TOKEN"])
r=c.predict(caption=None,image=handle_file("man_in2.png"),mv_image_front=None,mv_image_back=None,mv_image_left=None,mv_image_right=None,steps=50,guidance_scale=5,seed=1234,octree_resolution=200,check_box_rembg=False,num_chunks=8000,randomize_seed=True,api_name="/shape_generation")
import shutil;shutil.copy(r[0]["value"],"man_white4.glb");print(r[2]["number_of_faces"])
