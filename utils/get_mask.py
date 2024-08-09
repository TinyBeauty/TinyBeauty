import os
import cv2
import facer
import torch
import random
import numpy as np

def label_mask(array):
    unique_nonzero_elements = np.unique(array[array != 0])
    print(unique_nonzero_elements)
    label_array = np.zeros_like(array)
    new_labels = np.arange(1, 11)
    new_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 将非零元素映射到新标签
    for new_label in new_labels:
        label_array[array == new_label] = new_label
    return label_array

def get_mask(image_path, save_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    img = cv2.imread(image_path)
    w, h, c = img.shape
    image = cv2.resize(img, (1024, 1024))
    print(image.shape)
    image = facer.hwc2bchw(torch.tensor(image)).to(device=device)
    with torch.inference_mode():
        faces = face_detector(image)
        print(faces['rects'])
    with torch.inference_mode():
        faces = face_parser(image, faces) 
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)[0:1, :, :, :]  # nfaces x nclasses x h x w
    print(seg_probs.shape)
    vis_seg_probs = seg_probs.argmax(dim=1).float()
    vis_img = vis_seg_probs.sum(0, keepdim=True)
    print(image_path)
    mask = label_mask(vis_img.cpu().numpy())
    mask = cv2.GaussianBlur(mask[0], (11, 11), 0)
    print(mask.shape)
    cv2.imwrite(save_path, cv2.resize(mask, (h, w)))
    
if __name__ == "__main__":
    image_dir = "/mnt/vdb/qiaoqiao/AIData/Hard_test"
    save_dir = image_dir+"_mask_blur"
    os.makedirs(save_dir, exist_ok = True)
    
    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)
        save_path = os.path.join(save_dir, image_name)
        if image_name[0]=='.':continue
        get_mask(image_path, save_path)