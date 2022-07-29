import torchvision
import torch
from tqdm import tqdm
from timm.data.mixup import Mixup
import random
import matplotlib.pyplot as plt
import numpy as np
import math

def test(model, dataloader):
    c, t = 0, 0
    with torch.no_grad():
        model.eval()
        for images, labels in tqdm(dataloader):
            pred, _ = model(images.cuda())
            predidx = torch.argmax(pred, 1)
            c += torch.count_nonzero(predidx == labels.cuda())
            t += images.shape[0]
        model.train()
    return c,t


def norm(data):
    data = data-torch.min(data) if torch.min(data) < 0 else data
    data = data/torch.max(data) if torch.max(data) > 1 else data
    return data

def preview(images=[], labels=[],overlays=None):
    classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    nc = math.ceil(math.sqrt(len(images)))
    fig, ax = plt.subplots(ncols=nc, nrows=nc, constrained_layout=True)
    for i, image in enumerate(images):
        output = norm(image)
        assert torch.max(output) <= 1, 'fail'
        assert torch.min(output) >= 0, 'fail'
        target = (int(i/nc), i%nc)
        ax[target].set_title(classes[labels[i]])
        ax[target].imshow(output)
        if overlays is not None:
            ax[target].imshow(overlays[i], alpha=0.5)
    plt.show()
    plt.close()