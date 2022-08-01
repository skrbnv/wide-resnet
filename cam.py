from re import L
import torchvision
import torchvision.transforms as transforms
import torch
from libs.wrn import Wide_ResNet
#import torchinfo
import numpy as np
import random
import matplotlib.pyplot as plt
from  scipy import ndimage

torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

model_file = './checkpoints/wrn28x10_own_d0.3_re_ra_97.11.dict'
model = Wide_ResNet(28, 10, 0.3, 10).float().cuda() 
model.load_state_dict(torch.load(model_file))

transform = transforms.Compose([
    transforms.ToTensor(),
])
viewset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=False, transform=transform)
view_loader = torch.utils.data.DataLoader(viewset, batch_size=16, shuffle=False, num_workers=0)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
evalset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=False, transform=transform_test)
eval_loader = torch.utils.data.DataLoader(evalset, batch_size=16, shuffle=False, num_workers=0)
model.eval()



with torch.no_grad():
    for (images, labels),(views, _) in zip(eval_loader, view_loader):
        preds, convss = model(images.float().cuda())
        for image, view, pred, label, convs in zip(images,views,preds,labels,convss):
            if torch.argmax(pred) == label.cuda():
                ww = model.linear.weight[label]
                output = torch.mul(convs, ww.unsqueeze(1).unsqueeze(2).broadcast_to(convs.shape))
                cam = torch.sum(output, 0)
                cam = ndimage.zoom(cam.cpu(), 4, order=0)
                cam /= np.max(cam)
                mix = image.permute(1,2,0) - 1 + np.expand_dims(cam, 2)
                mix /= torch.max(mix)
                #fig = plt.figure()
                #im1 = plt.imshow(image.permute(2,1,0))
                fig, ax = plt.subplots(3)
                fig.suptitle(classes[label])
                ax[0].imshow(view.permute(1,2,0))
                ax[1].imshow(image.permute(1,2,0))
                ax[1].imshow(cam, alpha=.6)
                ax[2].imshow(mix)
                plt.show()
                plt.close()
        #print(preds)

