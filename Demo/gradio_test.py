import os
import cv2
import torch
import numpy as np
import gradio as gr
import torchvision.transforms as transforms
from REC import BeautyREC

params={'dim':24,
        'style_dim':48,
        'activ': 'relu',
        'n_downsample':2,
        'n_res':2,
        'pad_type':'reflect'
}

def process_frame(frame, img_size = 256):
    frame_ori = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame_ori, (img_size, img_size))
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    frame = transform(frame)
    frame_ori = transform(frame_ori)
    return [frame.unsqueeze(0), frame_ori.unsqueeze(0)]

def pred(input_img, style = "oumei"):
    if style=="style1":
        PATH = "./checkpoints/style1.pt"
    elif style=="style2":
        PATH = "./checkpoints/style2.pt"
    elif style=="style3":
        PATH = "./checkpoints/style3.pt"
    else:
        return None
    model = BeautyREC(params)
    model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu'))["cleaner"])
    model.eval()
    sample = process_frame(input_img, img_size = 256)
    pred = model(sample[0])
    frame_ori = sample[1][0].numpy().transpose([1, 2, 0])
    pred = pred[0].detach().cpu().numpy().transpose([1, 2, 0])
    pred = cv2.resize(pred, (input_img.shape[0], input_img.shape[1]))
    pred_face = pred + frame_ori
    pred_face = torch.clamp(torch.tensor(pred_face), -1, 1).numpy()
    pred_face = ((pred_face.copy()/2+0.5)*255).astype(np.uint8)
    pred_face = cv2.cvtColor(pred_face, cv2.COLOR_RGB2BGR)
    cv2.imwrite("test.png", pred_face[:,:,::-1])
    return pred_face[:,:,::-1]

demo = gr.Interface(fn = pred, 
                    inputs = [gr.Image(), "text"],
                    outputs = "image", 
                    examples = [["examples/01.png", "style1"], ["examples/03.png", "style2"]],
                    # 设置网页标题
                    title="Demo of TinyBeauty:Toward Tiny and High-quality Facial Makeup with Data Inflation Learning",
                    # 左上角的描述文字
                    description="Input a human face image (W*W, face region only) and input the makeup style you want.")
demo.launch(share=True, server_name="0.0.0.0", server_port=5200)