import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as dt2d
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm


__author__ = "__Girsh_Hegde__"


class Bouncer(nn.Module):
    """ Bouncer class
        author: girish d. hegde
        contact: girsh.dhc@gmail.com

    Args:
        input (np.ndarray): [h, w] - input binary image
        target (np.ndarray): [h, w] - target binary image
    """
    def __init__(self, input, target):
        super().__init__()
        dt, dtxy = dt2d(target == 0, return_indices=True)
        self.dtxy = torch.tensor(dtxy, dtype=torch.float32)

        y, x = np.where(input)
        self.x = nn.Parameter(torch.tensor(x, dtype=torch.float32))
        self.y = nn.Parameter(torch.tensor(y, dtype=torch.float32))
        self.loss = nn.MSELoss()
        self.optimizer = optim.SGD(
            [self.x, self.y],
            lr=1e1, momentum=0.997
        )

    def forward(self):
        self.optimizer.zero_grad()
        with torch.no_grad():
            h, w = self.dtxy.shape[1:]
            ty, tx = self.dtxy[
                :, self.y.data.clamp(0, h - 1).long(), 
                self.x.data.clamp(0, w - 1).long()
            ]
        loss = (self.loss(self.x, tx) + self.loss(self.y, ty))/2
        loss.backward()
        self.optimizer.step()
        return loss

    @torch.no_grad()
    def viz(self, delay=0):
        h, w = self.dtxy.shape[1:]
        canvas = torch.zeros(h, w)
        canvas[self.y.data.clamp(0, h - 1).long(), self.x.data.clamp(0, w - 1).long()] = 1.
        canvas = canvas.numpy()
        img = to_image(canvas, norm=True, save=None, show=True, delay=delay)
        return img


def to_image(
        img, norm=False, save=None, show=True,
        delay=0, rgb=True, bg=0,
    ):
    """ Function to show/save image 
        author: Girish D. Hegde - girish.dhc@gmail.com

    Args:
        img (np.ndarray): [h, w, ch] image(grayscale/rgb)
        norm (bool, optional): min-max normalize image. Defaults to False.
        save (str, optional): path to save image. Defaults to None.
        show (bool, optional): show image. Defaults to True.
        delay (int, optional): cv2 window delay. Defaults to 0.

    Returns:,
        (np.ndarray): [h, w, ch] - image.
    """
    if rgb:
        img = img[..., ::-1]
    if norm:
        img = (img - img.min())/(img.max() - img.min())
    if img.max() <= 1:
        img *=255
        img = img.astype(np.uint8)
    if save is not None:
        cv2.imwrite(save, img)
    if show:
        cv2.imshow('img', img)
        cv2.waitKey(delay)
        # cv2.destroyAllWindows()
    return img


def main():
    # target = 1 - cv2.imread('./data/wave.png', 0)/255
    target = np.zeros((800, 800))
    h, w, = target.shape
    cv2.circle(target, (w//2, h//2), min(w, h)//4, 1., 5)
    noise =  np.random.choice([0., 1.], size=(h, w), p=[0.99, 0.01])

    bouncer = Bouncer(noise, target)
    bar = tqdm.tqdm(range(1000))
    images = []
    for i in bar:
        loss = bouncer()
        img = bouncer.viz(1)
        images.append(img)
        bar.set_description("loss %.3f" %loss)
    imageio.mimsave('./data/circle.gif', images, fps=60)


if __name__ == '__main__':
    main()