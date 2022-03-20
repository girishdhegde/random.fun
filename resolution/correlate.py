import math

import numpy as np
import cv2
import imageio
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


__author__ = "__Girsh_Hegde__"
__ref__ = """
    https://twitter.com/matthen2/status/1483160741222006788
    https://www.pexels.com/video/bird-perched-on-the-hand-of-a-woman-10041604/
"""


def correlate(input, h_range=(200, 600), w_range=(400, 800), k=8):
    """ Function to find width and height of video using gradient
        author: girish d. hegde
        contact: girish.dhc@gmail.com

    Args:
        input (np.ndarray): input 1d sequence of video pixels.
        h_range (tuple[int]): (minh, maxh) height search space.
        w_range (tuple[int]): (minw, maxw) width search space.
        k (int): return top k resolutions.
    Returns:
        list[np.ndarray]: top k list of [t, h, w] output video frames.
        list[tuple]: top k list of resolutions (h, w, f).
        np.ndarray: heatmap of gradients at different resolutions.
    """
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    sobel = torch.tensor([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 2]
    ])
    temp = torch.zeros((1, 1, 3, 3, 3))
    temp[..., 0, :, :] = sobel
    temp[..., 2, :, :] = -sobel

    sobel = torch.tensor(temp, dtype=torch.float32, device=device)
    input = torch.tensor(input, dtype=torch.float32, device=device)

    seq_len = len(input)
    gradient_map = np.zeros((h_range[1] - h_range[0], w_range[1] - w_range[0]))
    for w in tqdm.tqdm(range(w_range[0], w_range[1])):
        for h in range(h_range[0], h_range[1]):
            t = math.ceil(seq_len / (h*w))
            padding = t*h*w - seq_len
            data = F.pad(input, [0, padding]).reshape(1, 1, -1, h, w)
            gradient = torch.sum((F.conv3d(data, sobel, bias=None)).absolute())
            gradient = float(gradient.item())
            gradient_map[h - h_range[0], w - w_range[0]] = gradient

    topk_h, topk_w = np.unravel_index(np.argsort(gradient_map.ravel())[:k], gradient_map.shape)
    topk_h += h_range[0]
    topk_w += w_range[0]

    print(f'\nMost probabal {k} resolutions of given video are:')
    videos = []
    resolutions = []
    for i, (height, width) in enumerate(zip(topk_h, topk_w)):
        frames = math.ceil(seq_len / (height*width))
        print(f'{i}. {height, width} with {frames} frames')

        padding = frames*height*width - seq_len
        video = F.pad(input, [0, padding]).reshape(-1, height, width)
        videos.append(video.cpu().numpy())
        resolutions.append((height, width, frames))

    return  videos, resolutions, gradient_map


def read_video(video, show=True, fps=None):
    frames = []
    cap = cv2.VideoCapture(video)
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape
            frame = cv2.resize(frame, (w//8, h//8))
            frames.append(frame)
            if show:
                cv2.imshow('Frame', frame)
                cv2.waitKey(1)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return np.stack(frames)


def main():
    video = './data/pexels-los-muertos-crew-10041604.mp4'

    frames = read_video(video)
    f, h, w = frames.shape
    print(f'Original video is of resolution {h, w} with {f} frames')
    input = frames.reshape(-1)
    print(f'Sequence converted from video is of length {len(input)}')

    videos, resolutions, gradient = correlate(input, h_range=(100, 150), w_range=(200, 250))
    gradient /= gradient.max()

    fig = plt.figure(figsize=(20, 20))
    plt.imshow(gradient, cmap='hot', interpolation='nearest')
    plt.savefig('./data/gradient.png')
    plt.show()

    maxh = max(h for h, w, f in resolutions)
    maxw = max(w for h, w, f in resolutions)
    maxf = max(f for h, w, f in resolutions)
    n = len(videos)
    grid = np.zeros((maxf, (n//4)*maxh, 4*maxw), dtype=np.uint8)
    for i, video in enumerate(videos):
        video = ((video/video.max())*255).astype(np.uint8)
        f, h, w = video.shape
        sw, sh = maxw*(i%4), maxh*(i//4)
        grid[:f, sh:sh + h, sw:sw + w] = video

    grid[..., ::maxh, :] = 255
    grid[..., :, ::maxw] = 255
    for frame in grid:
        cv2.imshow('frame', frame)
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    imageio.mimsave('./data/prediction.gif', grid, fps=30)

    return videos, resolutions, gradient


if __name__ == '__main__':
    main()
