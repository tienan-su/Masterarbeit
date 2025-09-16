#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 10:42:41 2023

@author: forootani
"""
import numpy as np
import sys
import os


def setting_directory(depth):
    current_dir = os.path.abspath(os.getcwd())
    root_dir = current_dir
    for i in range(depth):
        root_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
        sys.path.append(os.path.dirname(root_dir))
    return root_dir

root_dir = setting_directory(0)

from pathlib import Path
import torch
from scipy import linalg

import torch.nn as nn
import torch.nn.init as init


from tqdm import tqdm

import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import train_test_split
import warnings
import time


from siren_modules import Siren


warnings.filterwarnings("ignore")
np.random.seed(1234)
torch.manual_seed(7)
# CUDA support
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod


#######################################
#######################################    


class DeepSimulation(ABC):
    def __init__(self, ):            
        pass
    
    @abstractmethod
    def nn_models(self, ):
        pass
    
    @abstractmethod
    def optimizer_func(self, ):
        pass
    
    @abstractmethod
    def scheduler_setting(self,):
        pass
    pass

    
class WindDeepModel(DeepSimulation):
    def __init__(self,  in_features, out_features,
                 hidden_features_str, 
                 hidden_layers,  learning_rate_inr=1e-5
                 ):            
        super().__init__()
        self.in_features = in_features 
        self.out_features = out_features
        self.hidden_layers = hidden_layers
        self.hidden_features_str = hidden_features_str
        self.learning_rate_inr = learning_rate_inr
    
    
    def nn_models(self, ):
        
        # siren model initialization
        self.model_str_1 = Siren(
            self.in_features,
            self.hidden_features_str,
            self.hidden_layers,
            self.out_features,
            outermost_linear=True,
        ).to(device)


        
        models_list = [self.model_str_1, 
                       ]
        
        return models_list
        
    
    def optimizer_func(self, ):
        
        self.optim_adam = torch.optim.Adam(
            [
                {
                    "params": self.model_str_1.parameters(),
                    "lr": self.learning_rate_inr,
                    "weight_decay": 1e-6,
                },
               
                
            ]
        )
        
        return self.optim_adam
        
    
    def scheduler_setting(self):
            
        scheduler = torch.optim.lr_scheduler.CyclicLR(
                self.optim_adam,
                base_lr=0.1 * self.learning_rate_inr,
                max_lr=10 * self.learning_rate_inr,
                cycle_momentum=False,
                mode="exp_range",
                step_size_up=1000,
            )
            
        return scheduler
        
        
    def run(self):
            
        models_list = self.nn_models()
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()
        
        return models_list, optimizer, scheduler

        
################################################
################################################ 

import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.init as init
import torch_geometric
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.init as init

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels, bias=bias)
        # Remove the custom weight initialization to avoid the error
        # self.init_weights()

    # Remove init_weights method as it is not necessary with GCNConv
    # def init_weights(self):
    #    with torch.no_grad():
    #        init.xavier_uniform_(self.conv.weight)  # GCNConv doesn't have a `weight` attribute

    def forward(self, x, edge_index):
        return torch.relu(self.conv(x, edge_index))

class GNNDeepModel(DeepSimulation):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, learning_rate=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.learning_rate = learning_rate

    def nn_models(self):
        self.layers = nn.ModuleList()
        self.layers.append(GNNLayer(self.in_channels, self.hidden_channels))

        for _ in range(self.num_layers - 2):
            self.layers.append(GNNLayer(self.hidden_channels, self.hidden_channels))

        self.layers.append(GNNLayer(self.hidden_channels, self.out_channels))
        
        return self.layers

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x

    def optimizer_func(self):
        self.optim = torch.optim.Adam(self.layers.parameters(), lr=self.learning_rate)
        return self.optim

    def scheduler_setting(self):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=100, gamma=0.1)
        return scheduler
    
    def run(self):
        model_layers = self.nn_models()
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()
        
        return model_layers, optimizer, scheduler


"""
import torch
import torch_geometric
from torch_geometric.data import Data

# Example usage with GNNDeepModel

# Define hyperparameters
in_channels = 16
hidden_channels = 32
out_channels = 10
num_layers = 3
learning_rate = 1e-4

# Initialize the model
gnn_model = GNNDeepModel(in_channels, hidden_channels, out_channels, num_layers, learning_rate)

# Get the model layers, optimizer, and scheduler using the run method
model_layers, optimizer, scheduler = gnn_model.run()


# Example of creating data (you would replace this with your actual data)
x = torch.randn((100, in_channels))  # 100 nodes with in_channels features

# Correct usage of grid function, extracting edge_index and converting it to torch.long
height, width = 10, 10
edge_index, _ = torch_geometric.utils.grid(height, width)  # Get edge_index and ignore pos
edge_index = edge_index.to(torch.long)

# Forward pass through the model
output = gnn_model.forward(x, edge_index)
"""




############################################
############################################


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import time

class LSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    init.xavier_uniform_(param)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        return output, (hidden, cell)

class LSTMDeepModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, learning_rate=1e-5, learning_rate_inr=1e-5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.learning_rate_inr = learning_rate_inr

        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(LSTMLayer(self.input_size, self.hidden_size))
        for _ in range(self.num_layers - 1):
            self.lstm_layers.append(LSTMLayer(self.hidden_size, self.hidden_size))

        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        hidden_state = None
        cell_state = None
        for lstm_layer in self.lstm_layers:
            x, (hidden_state, cell_state) = lstm_layer(x)

        # Take the output from the last time step
        x = x[:, -1, :]
        x = self.fc(x)

        return x

    def optimizer_func(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate_inr)

    def scheduler_setting(self):
        # Setting up ReduceLROnPlateau scheduler
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_func(), 
            mode='min', 
            factor=0.1, 
            patience=10, 
            threshold=0.0001, 
            min_lr=1e-7
        )


    def run(self):
        model = self
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()

        return model, optimizer, scheduler




##########################################
##########################################

class CNN_BiLSTM(nn.Module):
    def __init__(self, input_features, cnn_filters, kernel_size,
                 lstm_hidden, lstm_layers, output_size, dropout,learning_rate=1e-5, learning_rate_inr=1e-5):
        super(CNN_BiLSTM, self).__init__()
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.learning_rate_inr = learning_rate_inr

        # 1D CNN Layer
        self.conv1d = nn.Conv1d(in_channels=input_features,
                                out_channels=cnn_filters,
                                kernel_size=kernel_size,
                                padding=kernel_size // 2)  # same padding

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        # BiLSTM Layer
        self.bilstm = nn.LSTM(input_size=cnn_filters,
                              hidden_size=lstm_hidden,
                              num_layers=lstm_layers,
                              batch_first=True,  # batch size is the first dimension
                              dropout=dropout if lstm_layers > 1 else 0,  # dropout only if there are multiple layers (prevents overfitting)
                              bidirectional=True)  # output size is 2*lstm_hidden

        # Fully connected layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(2 * lstm_hidden, output_size)  # 2 for bidirectional # make output size 1 for regression

    def forward(self, x):
        # Input shape: (batch, time, features)
        x = x.permute(0, 2, 1)  # to (batch, features, time) for Conv1d

        x = self.conv1d(x)       # -> (batch, cnn_filters, time)
        x = self.relu(x)
        x = self.pool(x)         # -> (batch, cnn_filters, time/2)

        x = x.permute(0, 2, 1)   # back to (batch, time, features) for LSTM

        output, _ = self.bilstm(x)  # -> (batch, time, 2*lstm_hidden)
                                    # output, (hn, cn) = self.bilstm(x) # hn is the hidden state of the last time step, cn is the cell state of the last time step
        x = output[:, -1, :]        # Take last time step

        x = self.dropout(x)
        out = self.fc(x)            # -> (batch, output_size)
        return out
 
        
    def optimizer_func(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate_inr)

    def scheduler_setting(self):
        # Setting up ReduceLROnPlateau scheduler
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_func(),
            mode='min',
            factor=0.1,
            patience=10,
            threshold=0.0001,
            min_lr=1e-7
        )


    def run(self):
        model = self
        optimizer = self.optimizer_func()
        scheduler = self.scheduler_setting()

        return model, optimizer, scheduler

