import torch
from tqdm import tqdm
from timm.data.mixup import Mixup
import matplotlib.pyplot as plt
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
    data = data-torch.min(data)
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

class MixupRun():
    def __init__(self, config) -> None:
        mixup_args = {
            'mixup_alpha': config['mixup_alpha'],
            'cutmix_alpha': config['cutmix_alpha'],
            'cutmix_minmax': config['cutmix_minmax'],
            'prob': config['mixup_prob'],
            'switch_prob': config['switch_prob'],
            'mode': config['mixup_mode'],
            'label_smoothing': config['label_smoothing'],
            'num_classes': config['classes']
            }
        self.mixup_fn = Mixup(**mixup_args)
    
    def __call__(self, inputs, labels):
        return self.mixup_fn(inputs, labels)