<img src="pictures_resnet50_simclr\transistor_28.png" width="700px"></img>

## Bootstrap Your Own Latent (BYOL), in Pytorch

[![PyPI version](https://badge.fury.io/py/byol-pytorch.svg)](https://badge.fury.io/py/byol-pytorch)

Practical implementation of an <a href="https://arxiv.org/abs/2006.07733">astoundingly simple method</a> for self-supervised learning that achieves a new state of the art (surpassing SimCLR) without contrastive learning and having to designate negative pairs.

This repository offers a module that one can easily wrap any image-based neural network (residual network, discriminator, policy network) to immediately start benefitting from unlabelled image data.

Update 1: There is now <a href="https://untitled-ai.github.io/understanding-self-supervised-contrastive-learning.html">new evidence</a> that batch normalization is key to making this technique work well

Update 2: A <a href="https://arxiv.org/abs/2010.10241">new paper</a> has successfully replaced batch norm with group norm + weight standardization, refuting that batch statistics are needed for BYOL to work

Update 3: Finally, we have <a href="https://arxiv.org/abs/2102.06810">some analysis</a> for why this works

<a href="https://www.youtube.com/watch?v=YPfUiOMYOEE">Yannic Kilcher's excellent explanation</a>

Now go save your organization from having to pay for labels :)

## Install

```bash
$ pip install byol-pytorch
```

## Usage

Simply plugin your neural network, specifying (1) the image dimensions as well as (2) the name (or index) of the hidden layer, whose output is used as the latent representation used for self-supervised training.

```python
import torch
from byol_pytorch import BYOL
from torchvision import models

resnet = models.resnet50(pretrained=True)

learner = BYOL(
    resnet,
    image_size = 256,
    hidden_layer = 'avgpool'
)

opt = torch.optim.Adam(learner.parameters(), lr=3e-4)

def sample_unlabelled_images():
    return torch.randn(20, 3, 256, 256)

for _ in range(100):
    images = sample_unlabelled_images()
    loss = learner(images)
    opt.zero_grad()
    loss.backward()
    opt.step()
    learner.update_moving_average() # update moving average of target encoder

# save your improved network
torch.save(resnet.state_dict(), './improved-net.pt')
```

That's pretty much it. After much training, the residual network should now perform better on its downstream supervised tasks.


## Citation

```bibtex

```