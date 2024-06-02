#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: util.py
# Created Date: Tuesday March 28th 2023
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Tuesday, 28th March 2023 1:32:41 am
# Modified By: Chen Xuanhong
# Copyright (c) 2023 Shanghai Jiao Tong University
#############################################################


import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import PIL
from PIL import Image

annotator_ckpts_path = os.path.join(os.path.dirname(__file__), 'ckpts')

class GramLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Detach target since we do not want to use it for gradient computation.

    def gram_matrix(self, input):
        n, c, w, h = input.shape
        features = input.view(n * c, w * h)
        G = torch.mm(features, features.t())
        return G.div(n * c * w * h)

    def forward(self, pred, target, mask):
        pred = pred * mask
        target = target * mask
        G = self.gram_matrix(pred)
        T = self.gram_matrix(target.detach()).detach()
        self.loss = F.l1_loss(G, T)
        return self.loss

def save_model_card(repo_name, images=None, base_model=str, prompt=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
instance_prompt: {prompt}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- lora
inference: true
---
    """
    model_card = f"""
# LoRA DreamBooth - {repo_name}

These are LoRA adaption weights for {base_model}. The weights were trained on {prompt} using [DreamBooth](https://dreambooth.github.io/). You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images
