import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from utils import get_edges_index, prepare_keypoints_image
import wandb
import numpy as np
'''
class GraphConvolution_att(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True, att_size=55, init_A=0):
        super(GraphConvolution_att, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(att_size, att_size))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        # AHW
        support = torch.matmul(input, self.weight)  # HW
        
        #print("att:", self.att.shape)
        #print("support: ",support.shape)
        #print("input: ",input.shape)
        #print("weight: ",self.weight.shape)
        
        output = torch.matmul(self.att, support)  # g
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
'''
class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):

        support = torch.matmul(input, self.weight)

        output = torch.matmul(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

class GraphEncoder(nn.Module):
    def __init__(self, in_features, num_points, hidden_features, out_features):
        super(GraphEncoder, self).__init__()
        self.conv1 = GraphConvolution(in_features, hidden_features)
        self.conv2 = GraphConvolution(hidden_features, out_features)
        self.conv3 = nn.Conv2d(num_points, 2, kernel_size=(1,1))
        self.num_points = num_points

    def forward(self, x, adj_m):
        x = F.relu(self.conv1(x, adj_m))
        x = self.conv2(x, adj_m)

        x = x.view(x.size(0), self.num_points, -1, 1)

        x = self.conv3(x)

        #x = torch.squeeze(x)

        return x

class GraphDecoder(nn.Module):
    def __init__(self, in_features, num_points, hidden_features, out_features):
        super(GraphDecoder, self).__init__()
        self.conv_2d = nn.Conv2d(out_features, num_points*in_features, kernel_size=(in_features,1), padding=(0,0))
        self.conv1 = GraphConvolution(in_features, hidden_features)
        self.conv2 = GraphConvolution(hidden_features, out_features)
        self.num_points = num_points

    def forward(self, x, adj_m):
        x = self.conv_2d(x)
        x = x.view(x.size(0), self.num_points,-1)
        x = F.relu(self.conv1(x, adj_m))
        x = self.conv2(x, adj_m)
        return x

class GraphAutoEncoder(nn.Module):
    def __init__(self, in_features, num_points, hidden_features, out_features):
        super(GraphAutoEncoder, self).__init__()
        self.encoder = GraphEncoder(in_features, num_points, hidden_features, out_features)
        self.decoder = GraphDecoder(out_features, num_points, hidden_features, in_features)
        
    def forward(self, x, adj_m):

        x = self.encoder(x, adj_m)
        x = self.decoder(x, adj_m)

        return x

######################################################3
# PYTORCH LIGHTNING MODULE

class LightningModule(pl.LightningModule):
    def __init__(self, in_features, num_points, hidden_features, out_features, lr, criterion):
        #super().__init__()
        super(LightningModule, self).__init__()
        self.save_hyperparameters()
        self.GAE = GraphAutoEncoder(in_features, num_points, hidden_features, out_features)
        self.lr = lr
        self.criterion = criterion

        self.connections = np.moveaxis(np.array(get_edges_index(num_points)), 0, 1)

    def forward(self, x, adj_m):
        x = self.GAE(x, adj_m)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        data, edges = train_batch
        data = data.float()
        edges = edges.float()

        z = self.GAE(data, edges)
        loss = self.criterion(z, data)
        return loss

    def validation_step(self, val_batch, batch_idx):

        data, edges = val_batch

        data = data.float()
        edges = edges.float()

        z = self.GAE(data, edges)
        loss = self.criterion(z, data)

        return {'input': data, 'output': z, 'loss': loss}


    def validation_epoch_end(self, validation_step_outputs):

        input_acum = validation_step_outputs[0]['input'].cpu().detach().numpy()
        output_acum = validation_step_outputs[0]['output'].cpu().detach().numpy()

        loss_acum = [tmp['loss'] for tmp in validation_step_outputs]

        avg_loss = sum(loss_acum)/len(loss_acum)

        self.log("val_loss_epoch", torch.tensor([avg_loss]))

        for pos, keypoints in enumerate(input_acum):
            if pos == 0:
                input_images = prepare_keypoints_image(keypoints, self.connections)
            if pos < 10:
                input_images = np.concatenate((input_images, prepare_keypoints_image(keypoints, self.connections)), axis=1)
            else:
                continue
        for pos, keypoints in enumerate(output_acum):
            if pos == 0:
                output_images = prepare_keypoints_image(keypoints, self.connections)
            if pos < 10:
                output_images = np.concatenate((output_images, prepare_keypoints_image(keypoints, self.connections)), axis=1)
            else:
                continue

        #for img_i, img_o in zip(input_images, output_images):
        output = np.concatenate((output_images, input_images), axis=0)
        images = wandb.Image(output, caption="Top: Output, Bottom: Input")
        wandb.log({"examples_validation epoch": images})


        







#model = GraphAutoEncoder(in_features=71*2, hidden_features=128, out_features=64)
