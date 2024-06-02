import os
import cv2
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from diffusers.utils import load_image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from diffusers_my.src.diffusers import ControlNetModel, MyStableDiffusionControlNetImg2ImgPipeline, \
    EulerAncestralDiscreteScheduler
    
from SD_model import ImageProjModel
from utils import resize_image, HWC3, get_mask_v2, blend_image_result


def parser_args():
    parser = argparse.ArgumentParser(description='Simple example of a testing script')

    parser.add_argument('--model_base', type=str, default='runwayml/stable-diffusion-v1-5',
                        help='Path to pretrained model from huggingface.co/models')
    parser.add_argument('--lora_path', '-m', type=str, default='./lora_makeup/purple_styleimage_no_mask',
                        help='The trained Lora model path')
    parser.add_argument('--style_path', '-s', type=str, default='data/Finetune_Data/purple.png',
                        help='Path of the style image')
    parser.add_argument('--data_root', '-d', type=str, default='data/Finetune_Data/test',
                        help='A folder containing the data of testing images')
    parser.add_argument('--output_dir', '-o', type=str, default='res/test1',
                        help='A folder containing the data of testing images')
    parser.add_argument('--denoising', type=float, default=0.3, help='Steps of denoising')
    parser.add_argument('--conditioning_scale', type=float, default=1.0, help='Conditioning scale')
    parser.add_argument('--resolution', type=int, default=512,
                        help='The resolution for input images, all the images will be resized to this')
    parser.add_argument('--image_num', type=int, default=20000, help='Image number')

    args = parser.parse_args()
    return args


def get_canny_image(image, resolution):
    image = resize_image(image, resolution)
    image = HWC3(image)
    return Image.fromarray(cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 100, 200))

def load_and_encode_image(image_path, clip_image_processor, image_encoder, image_proj_model):
    image = load_image(image_path)
    clip_image = clip_image_processor(images=image, return_tensors="pt").pixel_values
    clip_image_embeds = image_encoder(clip_image.to("cuda:0")).image_embeds
    image_prompt_embeds = image_proj_model(clip_image_embeds)
    uncond_image_prompt_embeds = image_proj_model(torch.zeros_like(clip_image_embeds))
    return image_prompt_embeds, uncond_image_prompt_embeds

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    controlnet = [
        ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16).to("cuda:0")
    ]
    generator = torch.manual_seed(10086)
    pipe_controlnet = MyStableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        args.model_base,
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16,
        generator=generator
    ).to("cuda:0")

    pipe_controlnet.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe_controlnet.scheduler.config)
    pipe_controlnet.enable_model_cpu_offload()
    pipe_controlnet.unet.load_attn_procs(args.lora_path)

    image_proj_model = ImageProjModel(
        cross_attention_dim=pipe_controlnet.unet.config.cross_attention_dim,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=32,
    ).to("cuda:0")

    image_proj_model.load_state_dict(torch.load(os.path.join(args.lora_path, "proj.pth")))
    clip_image_processor = CLIPImageProcessor()
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter",
                                                                  subfolder="models/image_encoder").to("cuda:0")

    for filename in tqdm(sorted(os.listdir(args.data_root))[:args.image_num]):
        if filename[0] == '.': continue
        image_path = os.path.join(args.data_root, filename)
        image = cv2.imread(image_path)
        if image is None:
            continue
        mask_path = os.path.join(args.data_root + '_mask', filename)
        save_path = os.path.join(args.output_dir, filename)

        canny_image = get_canny_image(image, args.resolution)
        mask = get_mask_v2(mask_path)

        image_prompt_embeds, uncond_image_prompt_embeds = load_and_encode_image(image_path, clip_image_processor,
                                                                                image_encoder, image_proj_model)
        style_image_prompt_embeds, uncond_style_image_prompt_embeds = load_and_encode_image(args.style_path,
                                                                                            clip_image_processor,
                                                                                            image_encoder,
                                                                                            image_proj_model)

        prompt_embeds = torch.cat([image_prompt_embeds, style_image_prompt_embeds], dim=1)
        negative_prompt_embeds = torch.cat([uncond_image_prompt_embeds, uncond_style_image_prompt_embeds], dim=1)

        image = load_image(image_path)
        W, H = image.size
        image = image.resize((args.resolution, args.resolution))

        # *********************** img2img **************************
        raw_image = pipe_controlnet(image=image,
                                    control_image=[canny_image],
                                    strength=args.denoising,
                                    mask=mask,
                                    prompt_embeds=prompt_embeds,
                                    negative_prompt_embeds=negative_prompt_embeds,
                                    guidance_scale=0.0,
                                    num_inference_steps=10,
                                    generator=generator,
                                    controlnet_conditioning_scale=args.conditioning_scale,
                                    cross_attention_kwargs={"scale": 0.8}
                                    ).images[0]

        res = pipe_controlnet(image=image,
                              control_image=[canny_image],
                              strength=args.denoising,
                              mask=mask,
                              prompt_embeds=prompt_embeds,
                              negative_prompt_embeds=negative_prompt_embeds,
                              guidance_scale=2.5,
                              num_inference_steps=10,
                              generator=generator,
                              controlnet_conditioning_scale=args.conditioning_scale,
                              cross_attention_kwargs={"scale": 0.9}
                              ).images[0]

        res.resize((int(W), int(H))).save(os.path.join(save_path))
        raw_image.resize((int(W), int(H))).save(os.path.join(save_path.replace('.png', '_raw.png')))
        detail_res = blend_image_result(raw_image, image, res, 0.7, mask['changed'].numpy())
        detail_res.save(os.path.join(save_path.replace('.png', '_detail.png')))


if __name__ == '__main__':
    args = parser_args()
    main(args)
