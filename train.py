from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from tqdm import tqdm
from libs.wrn import Wide_ResNet
from libs.functions import test, MixupRun
from libs.shedulers import StepDownScheduler
import wandb
import numpy as np
import random


config = {
    'seed': None, #None or int
    'initial_lr': 0.1,
    'depth': 28,
    'widen_factor': 10,
    'dropout': 0.3,
    'wandb': True,
    'resume': True,
    'project': 'cifar10-construction',
    'wandb_id': '1s2y72tl',
    'epoch': 81,
    'max_epochs': 300, 
    'model_file': './checkpoints/best.dict',
    'std': (0.4914, 0.4822, 0.4465),
    'mean': (0.2023, 0.1994, 0.2010),
    'classes': 10,
    'mixup_alpha': 0.8,
    'cutmix_alpha': 1.,
    'cutmix_minmax': None,
    'mixup_prob': 0.5,
    'switch_prob': 0.5,
    'mixup_mode': 'batch',
    'label_smoothing': 0,
    }

seed = random.randint(1, 5000) if config['seed'] is None else config['seed']
print(f'Using seed: {seed}')
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


if config['wandb']:
    if not config['resume']:
        wprj = wandb.init(project=config['project'], config={'seed': seed})
    else:
        wprj = wandb.init(id=config['wandb_id'],
                          project=config['project'],
                          resume="must")

model = Wide_ResNet(config['depth'], config['widen_factor'], config['dropout'], config['classes']).float().cuda()
if config['resume']:
    model.load_state_dict(torch.load(config['model_file']))
if config['wandb']:
    wandb.watch(model)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=3, magnitude=9),
    transforms.ToTensor(),
    transforms.Normalize(config['std'], config['mean']),
    transforms.RandomErasing(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(config['std'], config['mean']),
])

trainset = CIFAR10(root='./datasets', train=True, download=False, transform=transform_train)
trainset_test = torch.utils.data.Subset(CIFAR10(root='./datasets', train=True, download=False, transform=transform_test), random.sample(range(1, 50000), 10000))
evalset = CIFAR10(root='./datasets', train=False, download=False, transform=transform_test)

train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=0)
train_test_loader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=False, num_workers=0)
eval_loader = torch.utils.data.DataLoader(evalset, batch_size=64, shuffle=False, num_workers=0)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), config['initial_lr'], momentum=0.9, weight_decay=5e-4, nesterov=True)
scheduler = StepDownScheduler(optimizer, config['epoch'])
ev_acc_max = 0
mixup_fn = MixupRun(config)
for epoch in range(config['epoch'], config['max_epochs']):
    epoch_loss = []
    for images, labels in tqdm(train_loader):
        optimizer.zero_grad()
        images, labels = mixup_fn(images.cuda(), labels.cuda())
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
    #scheduler.step(c/t)
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
