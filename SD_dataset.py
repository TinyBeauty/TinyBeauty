import os
import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F

from PIL import Image
from insightface.app import FaceAnalysis
from torch.utils.data import Dataset
from torchvision import transforms
from torch.autograd import Variable


class Makeup_Text_Dataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
            self,
            instance_data_root,
            non_makeup_data_root,
            instance_prompt,
            tokenizer,
            size=512,
            num_data=None
    ):
        self.size = size
        self.num_data = num_data
        self.tokenizer = tokenizer
        self.dataset = []
        for f in sorted(os.listdir(non_makeup_data_root)):
            if f[0] == '.': continue
            img_path = os.path.join(instance_data_root, f)
            non_makeup_img_path = os.path.join(non_makeup_data_root, f)
            mask_img_path = os.path.join(non_makeup_data_root + '_mask', f)
            # if os.path.exists(img_path):
            self.dataset.append([img_path, non_makeup_img_path, mask_img_path, instance_prompt])
            # else:
            # continue
            # self.dataset.append([non_makeup_img_path, non_makeup_img_path, short_instance_prompt])
        self.num_instance_images = len(self.dataset)
        self._length = self.num_instance_images
        random.shuffle(self.dataset)
        self.image_transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        # ************** face parsing *****************
        # face:1, left_eyebrow:2, right_eyebrow:3, left_eye:4  right:5, nose:6, upper_lip:7, lower_lip:9 teeth:8 hair:10

        self.latent_size = 64
        self.lip_class = [7, 9]
        self.face_class = [1, 6, 11]
        self.eyes_class = [4, 5]
        self.eyebrows_class = [2, 3]
        self.styles = ["flower.png", "purple.png", "yuanying.png"]
        self.device = 'cpu'

    def __len__(self):
        if self.num_data is not None:
            return self.num_data
        else:
            return self._length

    def __getitem__(self, index):
        example = {}
        if len(self.styles) > 0:
            style = random.choice(self.styles)
            style_img = Image.open("data/Finetune_Data/" + style)
            example["style_images"] = style_img
            if style == "flower.png":
                hr_img = Image.open(self.dataset[index][0].replace("oumei", "flower"))
            elif style == "purple.png" or style == "purple_eye.png":
                hr_img = Image.open(self.dataset[index][0].replace("oumei", "purple"))
            elif style == "yuanying.png":
                hr_img = Image.open(self.dataset[index][0].replace("oumei", "yuanying"))
        else:
            style_img = None
            hr_img = Image.open(self.dataset[index][0])

        non_makeup_img = Image.open(self.dataset[index][1])
        if not hr_img.mode == "RGB":
            hr_img = hr_img.convert("RGB")
        hr_img = hr_img.resize((512, 512))
        example["content_images"] = non_makeup_img
        if not non_makeup_img.mode == "RGB":
            non_makeup_img = non_makeup_img.convert("RGB")
        non_makeup_img = non_makeup_img.resize((512, 512))
        if style_img is not None:
            if not style_img.mode == "RGB":
                style_img = style_img.convert("RGB")
            style_img = style_img.resize((512, 512))

        image = cv2.imread(self.dataset[index][1])
        ret = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        faces = self.app.get(ret)
        faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)

        hr_img = self.image_transforms(hr_img)
        example["instance_images"] = hr_img
        example["nonmakeup_images"] = self.image_transforms(non_makeup_img)
        if style_img is not None:
            example["makeup_only_images"] = self.image_transforms(style_img)
        example["instance_prompt_ids"] = self.tokenizer(
            self.dataset[index][3],
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        example["face_embeds"] = faceid_embeds

        mask = torch.tensor(cv2.imread(self.dataset[index][2], cv2.IMREAD_GRAYSCALE))

        if mask == None:
            print(self.dataset[index][2])
        mask = self.get_mask(mask)
        example["instance_mask"] = mask
        return example

    def mask_preprocess(self, mask_A, mask_B):
        index_tmp = torch.nonzero(mask_A, as_tuple=False)
        x_A_index = index_tmp[:, 2]

        y_A_index = index_tmp[:, 3]
        index_tmp = torch.nonzero(mask_B, as_tuple=False)
        x_B_index = index_tmp[:, 2]
        y_B_index = index_tmp[:, 3]
        index = [x_A_index, y_A_index, x_B_index, y_B_index]
        index_2 = [x_B_index, y_B_index, x_A_index, y_A_index]
        mask_A = mask_A.squeeze(0)
        mask_B = mask_B.squeeze(0)
        return mask_A, mask_B, index, index_2

    def to_var(self, x, requires_grad=True):
        if torch.cuda.is_available():
            x = x.cuda()
        if not requires_grad:
            return Variable(x, requires_grad=requires_grad)
        else:
            return Variable(x)

    def get_mask(self, mask):
        mask = F.interpolate(
            mask.view(1, 1, 1024, 1024),
            (self.latent_size, self.latent_size),
            mode="nearest")
        mask = mask.type(torch.uint8)
        mask = self.to_var(mask, requires_grad=False).to(self.device)

        # pdb.set_trace()
        mask_lip = (mask == self.lip_class[0]).float() + (mask == self.lip_class[1]).float()
        mask_face = (mask == self.face_class[0]).float() + (mask == self.face_class[1]).float()
        mask_eyebrow = (mask == self.eyebrows_class[0]).float() + (mask == self.eyebrows_class[1]).float()
        mask_eye_left = (mask == self.eyes_class[0]).float()
        mask_eye_right = (mask == self.eyes_class[1]).float()

        mask_eye_left, mask_eye_right = self.dilated_eyes(mask_eye_left[0], mask_eye_right[0])
        mask_eye_left, mask_eye_right = mask_eye_left.unsqueeze(0), mask_eye_right.unsqueeze(0)
        mask_eyes = mask_eye_left + mask_eye_right

        # mask_unchanged = 
        mask = {}
        mask["mask_eye"] = mask_eyes
        mask["mask_skin"] = mask_face
        mask["mask_lip"] = mask_lip
        # mask["unchanged"] = (mask == 0).float() + (mask == 10).float() + (mask == 11).float() + (mask == 12).float() + (mask == 13).float() + (mask == 8).float() + (mask == 4).float() + (mask == 5).float() 
        mask["changed"] = mask_face + mask_lip + mask_eyebrow
        return mask

    def dilated_eyes(self, mask_A, mask_B, save=False):
        # print(mask_A.shape)
        kernel = np.ones((3, 6), dtype=np.uint8)
        mask_A = np.array(mask_A[0]).astype(np.uint8)
        mask_B = np.array(mask_B[0]).astype(np.uint8)
        if save:
            mask_A[np.where(mask_A != 0)] = 255
            cv2.imwrite("testA.jpg", mask_A)
        dilated_mask_A = cv2.dilate(mask_A, kernel, iterations=4)
        dilated_mask_B = cv2.dilate(mask_B, kernel, iterations=4)
        # cv2.imwrite("testA_mask.jpg", dilated_mask_A-mask_A)
        return torch.from_numpy(dilated_mask_A - mask_A).unsqueeze(0), torch.from_numpy(
            dilated_mask_B - mask_B).unsqueeze(0)

