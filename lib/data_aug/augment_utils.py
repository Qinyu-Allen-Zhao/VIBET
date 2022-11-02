import math
import numpy as np
import torch
import torchvision.transforms.functional as F

def cut_augmentation(bbox):
    nframes, _ = bbox.shape
    center_offset = np.random.rand(2) - 0.5
    scale_offset = np.random.rand(2) - 0.5

    for i in range(nframes):
        c_x, c_y, w, h = bbox[i, :]
        c_x += c_x * center_offset[0]
        c_y += c_y * center_offset[1]
        w += w * scale_offset[0]
        h += h * scale_offset[1]

        bbox[i, :] = np.array([c_x, c_y, w, h])

    return bbox

def create_random_mask(images, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
    img = images[0]

    # Randomly generate one mask
    i, j, h, w, v = random_mask_for_one_image(img, ratio, scale, value)

    # Mask all images in the video segment
    for n, img in enumerate(images):
        images[n] = F.erase(img, i, j, h, w, v)

    # Return original image
    return images


def random_mask_for_one_image(img, ratio, scale, value):
    # Randomly create one mask and apply it to all frames in a video
    img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
    area = img_h * img_w
    log_ratio = torch.log(torch.tensor(ratio))
    for _ in range(10):
        erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        h = int(round(math.sqrt(erase_area * aspect_ratio)))
        w = int(round(math.sqrt(erase_area / aspect_ratio)))
        if not (h < img_h and w < img_w):
            continue

        if value is None:
            v = torch.empty([img_c, h, w], dtype=torch.float32).normal_()
        else:
            v = torch.tensor(value)[:, None, None]

        i = torch.randint(0, img_h - h + 1, size=(1,)).item()
        j = torch.randint(0, img_w - w + 1, size=(1,)).item()
        return i, j, h, w, v

    # Return original image
    return 0, 0, img_h, img_w, img