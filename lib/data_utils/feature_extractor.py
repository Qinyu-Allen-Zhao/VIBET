# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de
import math
import os
import torch
import torchvision
import torchvision.transforms.functional as F
import numpy as np
import matplotlib.pyplot as plt

from lib.utils.vis import batch_visualize_preds
from lib.data_utils.img_utils import get_single_image_crop, convert_cvimg_to_tensor


def extract_features(model, video, bbox, debug=False, batch_size=200,
                     kp_2d=None, dataset=None, scale=1.3, random_mask=False,
                     aug_features=False):
    '''
    :param model: pretrained HMR model, use lib/models/hmr.py:get_pretrained_hmr()
    :param video: video filename, torch.Tensor in shape (num_frames,W,H,C)
    :param bbox: bbox array in shape (T,4)
    :param debug: boolean, true if you want to debug HMR predictions
    :param batch_size: batch size for HMR input
    :param random_mask: Randomly mask some parts of the image to augment data
    :param aug_features: Augment features by combining features and SMPL parameters
    :return: features: resnet50 features np.ndarray -> shape (num_frames, 4)
    '''
    device = 'cuda'

    if isinstance(video, torch.Tensor) or isinstance(video, np.ndarray):
        video = video
    elif isinstance(video, str):
        if os.path.isfile(video):
            video, _, _ = torchvision.io.read_video(video)
        else:
            raise ValueError(f'{video} is not a valid file.')
    else:
        raise ValueError(f'Unknown type {type(video)} for video object')

    # For debugging ground truth 2d keypoints
    if debug and kp_2d is not None:
        import cv2
        if isinstance(video[0], np.str_):
            print(video[0])
            frame = cv2.cvtColor(cv2.imread(video[0]), cv2.COLOR_BGR2RGB)
        elif isinstance(video[0], np.ndarray):
            frame = video[0]
        else:
            frame = video[0].numpy()
        for i in range(kp_2d.shape[1]):
            frame = cv2.circle(
                frame.copy(),
                (int(kp_2d[0, i, 0]), int(kp_2d[0, i, 1])),
                thickness=3,
                color=(255, 0, 0),
                radius=3,
            )

        plt.imshow(frame)
        plt.show()

    if dataset == 'insta':
        video = torch.cat(
            [convert_cvimg_to_tensor(image).unsqueeze(0) for image in video], dim=0
        ).to(device)
    else:
        # crop bbox locations
        video = torch.cat(
            [get_single_image_crop(image, bbox, scale=scale).unsqueeze(0) for image, bbox in zip(video, bbox)], dim=0
        ).to(device)

    features = []

    # split video into batches of frames
    frames = torch.split(video, batch_size)

    with torch.no_grad():
        for images in frames:
            if random_mask:
                create_random_mask(images)

            if not debug:
                pred = model.aug_feature_extractor(images) \
                    if aug_features else model.feature_extractor(images)
                features.append(pred.cpu())
                del pred, images
            else:
                preds = model(images)
                dataset = 'spin'  # dataset if dataset else 'common'
                result_image = batch_visualize_preds(
                    images,
                    preds[-1],
                    target_exists=False,
                    max_images=4,
                    dataset=dataset,
                )

                plt.figure(figsize=(19.2, 10.8))
                plt.axis('off')
                plt.imshow(result_image)
                plt.show()

                del preds, images
                return 0

        features = torch.cat(features, dim=0)

    return features.numpy()


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