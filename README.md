# Wide ResNet fixer upper

## About 

Correct implementation (well, I hope so) of Wide ResNets based on <a href="https://github.com/meliketoy/wide-resnet.pytorch">meliketoy</a> code

Current best results with same augmentations as in original paper by S.Zagoruyko et al. <a href="https://arxiv.org/abs/1605.07146">PDF</a>:

Top1 Cifar10   |   Basic* | +Random Erase
--- | --- | ---
WRN 28x10 dropout=0.3 | 95.77% | - 
--- | --- | ---
WRN 28x10 no dropout | 95.93% | 96.77%

*Basic: Padding 4 & crop, random hor flip

### Installing

pip install -r requirements (It also uses W&B)
