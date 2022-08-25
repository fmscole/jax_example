import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from torch.nn import CTCLoss
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

# from torch.utils.tensorboard import SummaryWriter

import utils
from config import cfg

from PIL import Image
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

from generator import Generator
data_set = Generator(cfg.word.get_all_words(), 'horizontal')
train_sampler = torch.utils.data.RandomSampler(data_set)
data_loader = DataLoader(data_set, batch_size=128, sampler=train_sampler,
                             num_workers=4)
print(data_set.__getitem__(0))

for image, target, input_len, target_len in data_loader:
    print(input_len[0],target_len[0])
    image=image.numpy()
    maxpix=np.max(image)
    minpix=np.min(image)
    image=(image-minpix)/(maxpix-minpix)*255
    image=image[0][0]
    image=image.transpose(1,0)
    break

print(image.shape)
image=np.stack([image,image,image],axis=-1)
print(image.shape)
im=Image.fromarray(np.uint8(image))
im.save("t.jpeg")
plt.figure(1,figsize=(30,500))
plt.imshow(im)
plt.show()
print(target[0],target_len[0])
alpha=data_set.alpha
txt="".join([alpha[i] for i in target[0]])
print(txt,len(txt))