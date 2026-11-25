# ai-toolkit

Small toolkit to drive several AI models from one place: image to 3D (Hunyuan3D-2), image to video (LTX-Video, Wan 2.1 fast), plus helper scripts for cutouts, colouring a mesh from photos and rendering a spin video.

Libraries used: gradio_client, trimesh, numpy, scipy, Pillow, matplotlib, rembg.

Set `HF_TOKEN` (Hugging Face read token) before running. Free ZeroGPU quota is small (about 5 minutes of GPU per day), so runs can fail with a quota error.

Known limits: Hunyuan texture step and TRELLIS Spaces were failing when tested. Back and sides of a person from one photo are AI-invented.

`models.py` lists the services. `script_*.py` are the working scripts (they read the token from the HF_TOKEN env var).
