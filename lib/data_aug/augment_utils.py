import math
import numpy as np
import torch
import torchvision.transforms.functional as F


def cut_augmentation(raw_bbox):
    bbox = raw_bbox.copy()
    nframes, _ = bbox.shape
    center_offset = np.random.rand(2) * 0.5 - 0.25  # [-0.25, +0.25]
    center_offset[np.where(center_offset < 0)] -= 0.2
    center_offset[np.where(center_offset > 0)] += 0.2  # [-0.45, -0.2] U [0.2, 0.45]

    for i in range(nframes):
        c_x, c_y, w, h = raw_bbox[i, :]

        c_x += c_x * center_offset[0]
        c_y += c_y * center_offset[1]

        print(raw_bbox[i, :])
        print(c_x, c_y, w, h)
        print(center_offset)

        bbox[i, :] = np.array([c_x, c_y, w, h])

    return bbox


def create_random_mask(raw_images, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
    images = raw_images.clone()
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
            v = 0
        else:
            v = value

        i = torch.randint(0, img_h - h + 1, size=(1,)).item()
        j = torch.randint(0, img_w - w + 1, size=(1,)).item()
        return i, j, h, w, v

    # Return original image
    return 0, 0, img_h, img_w, img
