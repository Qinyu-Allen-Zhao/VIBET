import numpy as np


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
