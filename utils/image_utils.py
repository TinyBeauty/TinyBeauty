import cv2
import numpy as np
from PIL import Image

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def combine_l_ab(image_a, image_b):
    """
    Combine the L channel from image A with the ab channels from image B
    and return the combined image.
    
    Parameters:
    - image_a: The source image for the L channel (PIL Image object).
    - image_b: The source image for the ab channels (PIL Image object).
    
    Returns:
    - A PIL Image object with the combined channels.
    """

    # 首先检查两张图片的大小是否一致
    if image_a.size != image_b.size:
        raise ValueError("Images do not have the same dimensions.")

    # 将两张图像转换为LAB模式
    image_a_lab = image_a.convert('LAB')
    image_b_lab = image_b.convert('LAB')

    # 分离两张图像的通道
    l_channel, a_channel_a, b_channel_a = image_a_lab.split()
    l_channel_b, a_channel_b, b_channel_b = image_b_lab.split()

    # 将图片A的L通道与图片B的ab通道组合
    combined_lab = Image.merge('LAB', (l_channel * 0.5 + l_channel_b * 0.5, a_channel_b, b_channel_b))

    # 如果你想得到RGB模式的图片，可以将其转换回RGB
    combined_rgb = combined_lab.convert('RGB')

    return combined_rgb


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def create_gradient_image(img):
    height, width, c = img.shape
    gradient_image = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            green = int((y / height) * 255)
            red = int((x / width) * 255)

            gradient_image[y, x] = (0, green, red)

    return gradient_image

def blend_image_result(image_a, image_b, image_c, blend=1.0, mask=None):
    # 将图片转换为RGB模式，如果它们不是
    image_a = image_a.convert('RGB')
    image_b = image_b.convert('RGB')
    image_c = image_c.convert('RGB')

    # 计算残差并加到C上
    pixels_a = image_a.load()
    pixels_b = image_b.load()

    pixels_c = image_c.load()
    for y in range(image_a.size[1]):
        for x in range(image_a.size[0]):
            # 计算A和B的残差
            residual = tuple(pixels_b[x, y][i] - pixels_a[x, y][i] for i in range(3))
            residual_texture = tuple(pixels_c[x, y][i] - pixels_a[x, y][i] for i in range(3))
            # 将残差加到C上
            if mask is not None:
                new_pixel = tuple(min(255, max(0, pixels_a[x, y][i] + int(mask[y, x] * blend * residual[i]) + int(
                    mask[y, x] * 0.9 * residual_texture[i]))) for i in range(3))
            else:
                new_pixel = tuple(
                    min(255, max(0, pixels_a[x, y][i] + int(blend * residual[i]) + int(0.7 * residual_texture[i]))) for
                    i in range(3))
            pixels_c[x, y] = new_pixel
    return image_c

def Image_mul(img, mask):
    img = np.array(img)
    mask = np.array(mask)[0][0]
    return Image.fromarray(img * mask)

def Image_blend(imgA, maskA, imgB, maskB):
    imgA = np.array(imgA)
    maskA = np.array(maskA)
    imgB = np.array(imgB)
    maskB = np.array(maskB)
    img = imgA * maskA + imgB * maskB
    # print(np.max(img), np.min(img))
    # exit()
    return Image.fromarray(img.astype(np.uint8))

