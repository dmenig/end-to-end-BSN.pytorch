import os
import torch
from pretrainedmodels import inceptionv4, resnet18, pnasnet5large
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
import torchvision.models as models

import numpy as np

# EncoderCNN architecture
CNN_embed_dim = {}  # latent dim extracted by 2D CNN
CNN_embed_dim["inception"] = 1536
CNN_embed_dim["pnasnet"] = 1536
CNN_embed_dim["resnet18"] = 512
res_size = 224  # encoder image size
dropout_p = 0.0  # dropout probability

# DecoderRNN architecture
RNN_hidden_layers = 3
RNN_hidden_nodes = 512
RNN_FC_dim = 256


class CNNEncoder(nn.Module):
    def __init__(self, archi="inception", learnable_proportion=0.1):
        """Load the pretrained encoder and replace top fc layer."""
        super(CNNEncoder, self).__init__()

        if archi == "inception":
            encoder = inceptionv4(num_classes=1001, pretrained="imagenet+background")
        elif archi == "pnasnet":
            encoder = pnasnet5large(num_classes=1001, pretrained="imagenet+background")
        elif archi == "resnet18":
            encoder = resnet18(num_classes=1000, pretrained="imagenet")
        self.mean = encoder.mean
        self.std = encoder.std
        modules = list(encoder.children())[:-1]  # delete the last fc layer.
        ### freeze almost all layers
        num_learnable_layers = int(len(modules) * learnable_proportion)
        self.frozen_encoder = nn.Sequential(*modules[:-num_learnable_layers])
        self.learnable_encoder = nn.Sequential(*modules[-num_learnable_layers:])

    def forward(self, x):
        cnn_embed_seq = []
        B, C, T, H, W = x.size()
        # size B, C, T, H, W
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # size B, T, C, H, W
        x = x.view(B * T, C, H, W)
        # size B*T, C, H, W
        with torch.no_grad():
            x = self.frozen_encoder(x)  # encoder
        x = self.learnable_encoder(x)  # encoder
        # size B*T, C, 1, 1
        x = x.view(B, T, -1)  # flatten output of conv
        # size B, T, C
        return x
