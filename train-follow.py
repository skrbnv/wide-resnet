import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
#from torch.utils.data import DataLoader
from tqdm import tqdm
#from libs.model import Model
from libs.wrn import Wide_ResNet
from libs.functions import test, preview, norm
from libs.shedulers import StepDownScheduler
import wandb
import torchinfo
#import math
#import statistics
import numpy as np
import random
#from timm.data.dataset_factory import create_dataset
#from timm.data import create_loader
#from timm.data.mixup import Mixup
#from timm.data.transforms import RandomResizedCropAndInterpolation
from scipy import ndimage
#import matplotlib.pyplot as plt

config = {
    'seed': None, #None or inf
    'initial_lr': 1e-3,
    'depth': 28,
    'widen_factor': 10,
    'dropout': 0,
    'wandb': False,
    'project': 'cifar10-construction',
    'wandb_id': '2bzf9mlp',
    'resume': False,
    'epoch': 0,
    'max_epochs': 200, 
    'trainer_file': './checkpoints/wrn28x10_own_nodrop_9593.dict',
    'model_file': './checkpoints/checkpoint.dict',

}



seed = random.randint(1, 5000) if config['seed'] is None else config['seed']
print(f'Using seed: {seed}')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


trainer = Wide_ResNet(config['depth'], config['widen_factor'], config['dropout'], 10).float().cuda() 
trainer.load_state_dict(torch.load(config['trainer_file']))
trainer.eval()


if config['wandb']:
    if not config['resume']:
        wprj = wandb.init(project=config['project'], config={'seed': seed})
    else:
        wprj = wandb.init(id=config['wandb_id'],
                          project=config['project'],
                          resume="must")

#model = Model().float().cuda()
model = Wide_ResNet(config['depth'], config['widen_factor'], config['dropout'], 10).float().cuda() 
if config['resume']:
    model.load_state_dict(torch.load(config['model_file']))
if config['wandb']:
    wandb.watch(model)
torchinfo.summary(model, (32,3,32,32))


'''
mixup_args = {
    'mixup_alpha': 0.8,
    'cutmix_alpha': 1.,
    'cutmix_minmax': None,
    'prob': 0.5,
    'switch_prob': 0.5,
    'mode': 'batch',
    'label_smoothing': 0,
    'num_classes': 10}
mixup_fn = Mixup(**mixup_args)
'''

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(root='./datasets', train=True, download=False, transform=transform_train)
trainset_test = torch.utils.data.Subset(torchvision.datasets.CIFAR10(root='./datasets', train=True, download=False, transform=transform_test), random.sample(range(1, 50000), 10000))
evalset = torchvision.datasets.CIFAR10(root='./datasets', train=False, download=False, transform=transform_test)


train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=0)
train_test_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=0)
eval_loader = torch.utils.data.DataLoader(evalset, batch_size=16, shuffle=False, num_workers=0)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=config['initial_lr'], momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = StepDownScheduler(optimizer, config['epoch'])
ev_acc_max = 0
for epoch in range(config['epoch'], config['max_epochs']):
    epoch_loss = []

    for images, labels in tqdm(eval_loader):
        optimizer.zero_grad()
        #images, labels = mixup_fn(images, labels)
        with torch.no_grad():
            preds, convs = trainer(images.cuda())
            cams = []
            for i in range(images.shape[0]):
                ww = trainer.linear.weight[labels[i].item()]
                output = torch.mul(ww.unsqueeze(1).unsqueeze(2).broadcast_to(convs[i].shape), convs[i])
                cam = torch.sum(output, 0)
                cam = torch.from_numpy(ndimage.zoom(cam.cpu(), 4,order=0))
                cam = norm(cam)
                cam = cam.unsqueeze(2)
                #mix = image.permute(1,2,0)
                #mix /= torch.max(mix)
                #noise = cam * torch.rand(mix.shape)
                #noised = norm(image.permute(1,2,0)-1+noise)
                print(f'Pred: {torch.argmax(preds[i])}, actual: {labels[i]}')
                #preview([previews[i].permute(1,2,0),images[i].permute(1,2,0), cam], labels[i], {'image': cam, 'target': 1})
                cams.append(cam)
                #images[i] = mix.permute(2,0,1)
            preview(images.permute(0,2,3,1), labels, cams)

        preds, _ = model(images.cuda())
        loss = criterion(preds, labels.cuda())
        loss.backward()
        optimizer.step()
        epoch_loss.append(loss.item())

    c_ev,t_ev = test(model, eval_loader)
    c_tr,t_tr = test(model, train_test_loader)
    ev_acc = c_ev*100/t_ev
    if ev_acc > ev_acc_max:
        ev_acc_max = ev_acc
        torch.save(model.state_dict(), f'./checkpoints/best.dict')
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print(f'Epoch: {epoch}')
    print(f'    Learning rate: {lr:.4f}')
    print(f'    Training loss: {sum(epoch_loss)/len(epoch_loss):.4f}')
    print(f'    Training accuracy: {c_tr*100/t_tr:.2f}%')
    print(f'    Evaluation accuracy: {ev_acc:.2f}%')
    print(f'    Max evaluation accuracy: {ev_acc_max:.2f}%')

    if config['wandb']:
        wandb.log({
            "Train acc": c_tr*100/t_tr,
            "Eval acc": c_ev*100/t_ev,
            "Learning rate": optimizer.param_groups[0]['lr'],
            "Loss": sum(epoch_loss)/len(epoch_loss)
        })
