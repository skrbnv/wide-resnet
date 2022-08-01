# Wide ResNet 

## About 

Correct implementation (well, I hope I didn't miss anything) of Wide ResNets based on <a href="https://github.com/meliketoy/wide-resnet.pytorch">meliketoy</a> code.

Current best results with same augmentations as in original paper by S.Zagoruyko et al. <a href="https://arxiv.org/abs/1605.07146">(PDF)</a>:

Top1 Cifar10   |   Basic* | +Random Erase | +RE+RandAugment | +RE+RA+CutMix 
--- | --- | --- | --- | ---
WRN 28x10 dropout=0.3 | 95.77% | 96.84% | 97.27% | 97.28%

*Basic: Padding 4 & crop, random horizontal flip

### Installing

pip install -r requirements (It also uses W&B)
