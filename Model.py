%pip install timm

#Visualise multiple bands
from glob import glob
%pip install earthpy
import earthpy as et
import earthpy.spatial as es
import earthpy.plot as ep
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

%pip install plotly
import plotly.graph_objects as go
import os
from rasterio.enums import Resampling


%pip install seaborn
import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms as T
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
%pip install timm
import timm
from timm.loss import LabelSmoothingCrossEntropy # This is better than normal nn.CrossEntropyLoss

import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
%matplotlib inline
import sys


import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision import models
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split




%pip install --quiet pytorch-lightning>=1.4


import random
import seaborn as sns
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import read_image
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
%matplotlib inline

import math
import json
from functools import partial
from PIL import Image
import time

## Imports for plotting
import matplotlib.pyplot as plt
plt.set_cmap('cividis')
%matplotlib inline

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.reset_orig()

## tqdm for loading bars
from tqdm.notebook import tqdm

## PyTorch

import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision import transforms

# Import tensorboard
%load_ext tensorboard
# PyTorch Lightning
try:
    import pytorch_lightning as pl
except ModuleNotFoundError: # Google Colab does not have PyTorch Lightning installed by default. Hence, we do it here if necessary

    import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

from google.colab import drive
drive.mount('/content/drive')

#dataset_size = 6000


def img_to_patch(x, patch_size, flatten_channels=False):
   
    B, C, H, W = x.shape
    x = x.reshape(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5) # [B, H', W', C, p_H, p_W]
    x = x.flatten(1,2)              # [B, H'*W', C, p_H, p_W]
    x = x.permute(0,2,1,3,4)          # [B, C, H'*W', p_H, p_W]
    x = x.flatten(3,4)
    if flatten_channels:
        x = x.flatten(2,4)          # [B, H'*W', C*p_H*p_W]
    return x

class AttentionBlock(nn.Module):

    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        """
        Inputs:
            embed_dim - Dimensionality of input and attention feature vectors
            hidden_dim - Dimensionality of hidden layer in feed-forward network
                         (usually 2-4x larger than embed_dim)
            num_heads - Number of heads to use in the Multi-Head Attention block
            dropout - Amount of dropout to apply in the feed-forward network
        """
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(264)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                          dropout=dropout)
        #self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(264, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 264),
            nn.Dropout(dropout)
        )


    def forward(self, x):
        #print("In attn",x.shape)
        inp_x = self.layer_norm_1(x)
        #print("In attn",x.shape)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        #print("In attn",x.shape)
        #x = x + self.linear(x)
        #print("In attn",x.shape)
        return x

class VisionTransformer(nn.Module):

    def __init__(self, embed_dim=256, hidden_dim=512, num_channels=15, num_heads=1, num_layers=1, num_classes=6, patch_size=12, num_patches=36, dropout=0.2):
        """
        Inputs:
            embed_dim - Dimensionality of the input feature vectors to the Transformer
            hidden_dim - Dimensionality of the hidden layer in the feed-forward networks
                         within the Transformer
            num_channels - Number of channels of the input (3 for RGB)
            num_heads - Number of heads to use in the Multi-Head Attention block
            num_layers - Number of layers to use in the Transformer
            num_classes - Number of classes to predict
            patch_size - Number of pixels that the patches have per dimension
            num_patches - Maximum number of patches an image can have
            dropout - Amount of dropout to apply in the feed-forward network and
                      on the input encoding
        """
        super().__init__()

        self.patch_size = patch_size

        # Layers/Networks
        #self.input_layer_0 = nn.Linear(num_channels, 3)
        self.conv1_1 = nn.Conv2d(in_channels=15, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)


        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Iconv1 =  nn.Conv2d(256,1, kernel_size=1)

        self.Iconv2 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.Conv2d(32,1,3,padding=1)
        )
        self.Iconv3 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=1),
            nn.Conv2d(32,32,3,padding=1),
            nn.Conv2d(32,1,3,padding=1)
        )
        #self.input_layer = nn.Linear(num_channels, embed_dim)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.conv1 = nn.Conv2d(embed_dim+3,264,3,stride=1,padding=1)
        self.transformer = nn.Sequential(*[AttentionBlock(264, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers)])

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(4224),
            nn.Linear(4224, 81),
            nn.Linear(81, num_classes)
        )
        self.dropout = nn.Dropout(dropout)


        # Parameters/Embeddings
        self.cls_token = nn.Parameter(torch.randn(1,1,1,embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1,1+(num_patches),self.patch_size*self.patch_size,embed_dim))


    def forward(self, x):
        # Preprocess input
        #print("In ViT",x.shape)
        #print(x.shape)
        '''plt.figure()
        plt.axis('off')
        plt.imshow(show_rgb(x[0].detach()))'''
        #VGG
        x = (self.conv1_1(x))

        x = (self.conv2_1(x))
        '''plt.figure()
        plt.axis('off')
        plt.imshow(x[0][3].detach())'''
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        x = F.relu(self.conv3_1(x))
        x = (self.conv3_2(x))
        '''plt.figure()
        plt.axis('off')
        plt.imshow(x[0][3].detach())'''
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        #print("Done vgg 256*8*8",x.shape)

        #Inception
        X=self.maxpool(F.relu(self.Iconv1(x)))
        #print("1",X.shape)
        Y=self.maxpool(F.relu(self.Iconv2(x)))
        #print("2",Y.shape)
        Z=self.maxpool(F.relu(self.Iconv3(x)))
        #print("3",Z.shape)
        W= self.maxpool(x)
        #print("4",X.shape)
        inceptionout = [X,Y,Z,W]
        x = torch.cat(inceptionout,1)
        #print("Done inception",x.shape)
        #x = img_to_patch(x, self.patch_size)
        #print("Patches ",x.shape)
        '''plt.figure()
        plt.axis('off')
        plt.imshow(x[0][3].detach())'''

        x = self.dropout(x)

        x = F.relu(self.conv1(x))
        #print(x.shape)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])

        #print("Pool ",x.shape)
        x = x.permute(0,2,1)
        x = self.transformer(x)
        #print("Transformer out",x.shape)

        # Perform classification prediction
        #cls = x[:,0,:]
        x = x.flatten()
        #print(x.shape)
        out = self.mlp_head(x)
        #print("final out",out)
        out = out.unsqueeze(0)
        return out

from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix

class ViT(pl.LightningModule):

    def __init__(self, model_kwargs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.model = VisionTransformer(**model_kwargs)
        self.example_input_array = next(iter(train_loader))[0]
        self.val_step_outputs = []        # save outputs in each batch to compute metric overall epoch
        self.val_step_targets = []

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150], gamma=0.1)
        #return [optimizer], [lr_scheduler]
        return [optimizer]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        total = 27629
        #weights = [total/7209,total/9237,total/1698,total/1808,total/3890,total/3787]

        #loss = F.cross_entropy(preds, labels)
        #print("out shape", preds.shape, preds)
        label_one_hot = torch.nn.functional.one_hot(labels.type(torch.int64), num_classes = len(classes))
        loss = F.cross_entropy(preds[0], label_one_hot.type(torch.float32)[0].to(device))
        acc = (preds.argmax(dim=-1) == labels).float().mean()
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        if mode=='test':
          self.val_step_outputs.append(torch.tensor(preds.argmax(dim=-1), dtype=torch.int8))
          self.val_step_targets.append(torch.tensor(labels, dtype=torch.int8))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")

    def on_test_epoch_end(self):
        val_all_outputs = torch.stack(self.val_step_outputs)
        val_all_targets = torch.stack(self.val_step_targets)
        #print(len(val_all_outputs), len(val_all_targets))
        f1_macro_epoch = f1_score(val_all_outputs.cpu(), val_all_targets.cpu(),average='macro')
        precision_macro_epoch = precision_score(val_all_outputs.cpu(), val_all_targets.cpu(),average='macro')
        recall_macro_epoch = recall_score(val_all_outputs.cpu(), val_all_targets.cpu(),average='macro')
        cm_macro_epoch = confusion_matrix(val_all_outputs.cpu(), val_all_targets.cpu())
        print("Confusion Matrix", cm_macro_epoch)
        self.log(f'F1', f1_macro_epoch)
        self.log(f'Precision', precision_macro_epoch)
        self.log(f'Recall', recall_macro_epoch)
        self.val_step_outputs.clear()
        self.val_step_targets.clear()


def train_model(**kwargs):
    trainer = pl.Trainer(default_root_dir=os.path.join(CHECKPOINT_PATH, "ViT"),
                         max_epochs = 180,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")])
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = True # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training

    #ckpt_file = os.listdir("/content/drive/MyDrive/Colab Notebooks/ViT/lightning_logs/version_77/checkpoints/")[0]
    #pretrained_filename = os.path.join("/content/drive/MyDrive/Colab Notebooks/ViT/lightning_logs/version_77/checkpoints/", ckpt_file)
    pretrained_filename = "/content/drive/MyDrive/Colab Notebooks/ViT/lightning_logs/Acc_69.92/checkpoints/epoch=5-step=56016.ckpt"
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, Retraining....")
        model = ViT.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(42) # To be reproducable
        print("Training from scratch")
        model = ViT(**kwargs)

    trainer.fit(model, train_loader, val_loader)
    model = ViT.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    #train_result = trainer.test(model, train_loader, verbose=True)
    val_result = trainer.test(model, val_loader, verbose=True)

    #test_result = trainer.test(model, test_loader, verbose=False)
    result = {"val": val_result[0]["test_acc"]}
    test_result = trainer.test(model, test_loader, verbose=True)
    print(test_result)

    return model, result


def show_rgb(msi):
  return torch.stack((msi[4],msi[3],msi[2]),dim=2)/torch.max(torch.max(torch.max(msi[3]),torch.max(msi[2])),torch.max(msi[1]))



CHECKPOINT_PATH="/content/drive/MyDrive/Colab Notebooks/"
model, results = train_model(model_kwargs={
                                'embed_dim': 256,
                                'hidden_dim': 512,
                                'num_heads': 1,
                                'num_layers': 1,
                                'patch_size': 2,
                                'num_channels': 15,
                                'num_patches': 36*36,
                                'num_classes': len(classes),
                                'dropout': 0.4
                            },
                            lr=3e-4)
print("ViT results", results)




X_train,Y_train=load_tset1()
X_val,Y_val=load_valset1()
X_train=np.array(X_train)
Y_train=np.array(Y_train)
X_val=np.array(X_val)
Y_val=np.array(Y_val)
print(X_train.shape)
print(Y_train.shape)
print(X_val.shape)
print(Y_val.shape)

y_ser=pd.Series(Y_train)
print("Training Data:")
print(y_ser.value_counts())
print("Testing Data:")
y_sr=pd.Series(Y_val)
print(y_sr.value_counts())

model.model.conv1

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook



