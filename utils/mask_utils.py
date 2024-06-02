import cv2
import torch
import numpy as np
import torch.nn as nn


def get_mask_v2(mask_path):
    mask = torch.tensor(cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (512, 512))).type(
        torch.uint8).unsqueeze(2)

    latent_size = 64
    lip_class = [7, 9]
    face_class = [1]
    eyes_class = [4, 5]
    eyebrows_class = [2, 3]
    mask_lip = (mask == lip_class[0]).float() + (mask == lip_class[1]).float()
    mask_bg = (mask == 0).float() + (mask == 10).float() + (mask == 11).float() + (mask == 12).float() + (
                mask == 13).float()
    mask_face = (mask == face_class[0]).float() + (mask == 6).float()
    mask_eye_left = (mask == eyes_class[0]).float()
    mask_eye_right = (mask == eyes_class[1]).float()
    mask_eyeball = mask_eye_left + mask_eye_right
    mask_eyebrow = (mask == eyebrows_class[0]).float() + (mask == eyebrows_class[1]).float()
    mask_eye_left, mask_eye_right = dilated_eyes(mask_eye_left, mask_eye_right)
    mask_eyes = mask_eye_left + mask_eye_right
    mask_teeth = (mask == 8).float()
    mask_others = (mask == 8).float() + (mask == 4).float() + (mask == 5).float()
    kernel = np.ones((6, 6), np.uint8)
    cleaned_mask = cv2.morphologyEx(np.array(mask_others), cv2.MORPH_OPEN, kernel)
    mask_changed = mask_face + mask_eyebrow + mask_lip + mask_others
    mask = {}
    mask["mask_eye"] = mask_eyes[:, :, 0].clamp(0, 1)
    # mask["mask_skin"] = mask_changed[:, :, 0]
    mask["mask_lip"] = (mask_lip + mask_teeth)[:, :, 0]
    mask["unchanged"] = nn.functional.interpolate(mask_bg.squeeze(2).unsqueeze(0).unsqueeze(0),
                                                  (latent_size, latent_size))
    mask["changed"] = torch.tensor(cv2.GaussianBlur(np.array(mask_changed), (5, 5), 0))
    mask["others"] = torch.tensor(cleaned_mask).unsqueeze(2)
    mask["soft_mask"] = np.expand_dims(cv2.GaussianBlur(np.array(mask_others), (5, 5), 0), axis=2)
    return mask


def get_mask_v3(mask_path):
    mask_512 = torch.tensor(cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (512, 512))).type(
        torch.uint8).unsqueeze(2)
    mask = torch.tensor(cv2.resize(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE), (64, 64))).type(torch.uint8).unsqueeze(
        2)

    latent_size = 64
    lip_class = [7, 9]
    face_class = [1]
    eyes_class = [4, 5]
    eyebrows_class = [2, 3]
    kernel = np.ones((2, 2), np.uint8)
    # pdb.set_trace()
    mask_lip = (mask == lip_class[0]).float() + (mask == lip_class[1]).float()
    mask_bg = (mask == 0).float() + (mask == 10).float() + (mask == 11).float() + (mask == 12).float() + (
                mask == 13).float()
    mask_face = (mask == face_class[0]).float() + (mask == 6).float()
    mask_eye_left = (mask == eyes_class[0]).float()
    mask_eye_right = (mask == eyes_class[1]).float()
    mask_eyeball = mask_eye_left + mask_eye_right
    mask_eyebrow = (mask == eyebrows_class[0]).float() + (mask == eyebrows_class[1]).float()
    cleaned_eye_left = cv2.morphologyEx(np.array(mask_eye_left), cv2.MORPH_OPEN, kernel)
    cleaned_eye_right = cv2.morphologyEx(np.array(mask_eye_right), cv2.MORPH_OPEN, kernel)
    mask_eye_left, mask_eye_right = dilated_eyes(cleaned_eye_left, cleaned_eye_right)

    mask_eyes = mask_eye_left + mask_eye_right
    mask_teeth = (mask == 8).float()
    mask_others = (mask_512 == 4).float() + (mask_512 == 5).float()
    # mask_bg = mask_bga
    mask_changed = mask_face + mask_eyebrow
    mask = {}
    cleaned_mask = cv2.morphologyEx(np.array(mask_others), cv2.MORPH_OPEN, kernel)

    mask["mask_eye"] = mask_eyes[:, :, 0].clamp(0, 0.8)
    mask["mask_skin"] = mask_changed[:, :, 0]
    mask["mask_lip"] = (mask_lip + mask_teeth)[:, :, 0]
    mask["unchanged"] = nn.functional.interpolate(mask_bg.squeeze(2).unsqueeze(0).unsqueeze(0),
                                                  (latent_size, latent_size))
    mask["changed"] = mask_changed.squeeze(2)
    mask["others"] = torch.tensor(cleaned_mask).unsqueeze(2)
    mask["soft_mask"] = np.expand_dims(cv2.GaussianBlur(np.array(mask_others), (5, 5), 0), axis=2)

    return mask


def dilated_eyes(mask_A, mask_B, save=True):
    kernel = np.ones((2, 3), dtype=np.uint8)
    mask_A = np.array(mask_A).astype(np.uint8)
    mask_B = np.array(mask_B).astype(np.uint8)
    # if save:
    #     mask_A[np.where(mask_A!=0)]=255
    #     cv2.imwrite("testA.jpg", mask_A)
    dilated_mask_A = cv2.dilate(mask_A, kernel, iterations=2)
    dilated_mask_B = cv2.dilate(mask_B, kernel, iterations=2)
    shifted_mask_A = np.zeros_like(dilated_mask_A)
    shifted_mask_A[:-3, :] = dilated_mask_A[3:, :]
    shifted_mask_B = np.zeros_like(dilated_mask_B)
    shifted_mask_B[:-3, :] = dilated_mask_B[3:, :]
    return torch.from_numpy(shifted_mask_A).unsqueeze(2), torch.from_numpy(shifted_mask_B).unsqueeze(2)


def erode_teeth(mask_teeth):
    # print(mask_A.shape)
    kernel = np.ones((1, 2), dtype=np.uint8)
    mask_teeth = np.array(mask_teeth[0]).astype(np.uint8)
    dilated_mask_teeth = cv2.erode(mask_teeth, kernel, iterations=1)
    return torch.from_numpy(dilated_mask_teeth).unsqueeze(0)
