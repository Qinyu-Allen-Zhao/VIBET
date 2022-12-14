import math
import numpy as np
import torch
import torchvision.transforms.functional as F


def cut_augmentation(raw_bbox, debug=False):
    """
    Greate random cutting boounding box from the video segment
    """
    bbox = raw_bbox.copy()
    nframes, _ = bbox.shape

    ##################################################################
    # STEP 1 - Generate two random decimal offset factor between -0.25 and 0.25 and  
    #  remap them to -0.45 to -0.2 or 0.2 to 0.45.
    ##################################################################

    center_offset = np.random.rand(2) * 0.5 - 0.25  # [-0.25, +0.25]
    center_offset[np.where(center_offset < 0)] -= 0.2
    center_offset[np.where(center_offset > 0)] += 0.2  # [-0.45, -0.2] U [0.2, 0.45]


    ##################################################################
    # STEP 2 - Add an offset to the original center coordinates 
    #  obtained by multiplying the center coordinates with the offset 
    #  factor.
    ##################################################################
    for i in range(nframes):
        c_x, c_y, w, h = raw_bbox[i, :]

        c_x += c_x * center_offset[0]
        c_y += c_y * center_offset[1]

        if debug:
            print(raw_bbox[i, :])
            print(c_x, c_y, w, h)
            print(center_offset)

        bbox[i, :] = np.array([c_x, c_y, w, h])

    return bbox


def create_random_mask(raw_images, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
    """
    Greate random mask for the video segment from one frame
    """
    images = raw_images.clone()
    img = images[0]

    ##################################################################
    # STEP 1 - Randomly generate one mask from one frame
    ##################################################################
    i, j, h, w, v = random_mask_for_one_image(img, ratio, scale, value)

    ##################################################################
    # STEP 2 - Mask all images in the video segment with same mask.
    ##################################################################
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
        ##################################################################
        # STEP 1 - Randomly generate the area to be masked
        ##################################################################
        erase_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
        aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

        ##################################################################
        # STEP 2 - Generate the length and width based on the area until 
        #  it matches the image size
        ##################################################################
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
