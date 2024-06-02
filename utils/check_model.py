import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline, EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline
from transformers import CLIPVisionModelWithProjection

controlnet = [
            ControlNetModel.from_pretrained("lllyasviel/control_v11f1e_sd15_tile", torch_dtype=torch.float16),  
            ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),   
            ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-normal", torch_dtype=torch.float16) 
]
pipe_controlnet = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
        generator=None
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder = "models/image_encoder")
