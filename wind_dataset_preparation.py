#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 10:20:57 2024

@author: forootan
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


root_dir = setting_directory(1)

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

warnings.filterwarnings("ignore")

np.random.seed(1234)
torch.manual_seed(7)
#CUDA support


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

from abc import ABC, abstractmethod






#############################

class DataPreparation(ABC):
    def __init__(self, coords, data):
        self.data = data
        self.coords = coords

    @abstractmethod
    def prepare_data_random(self, test_data_size):
        pass
    
    
    #@abstractmethod
    #def prepare_data_DEIM(self):
    #    pass

#####################################################
#####################################################


class WindDataGen(DataPreparation):
    def __init__(self, coords, data, noise_level = None):
        
        
        super().__init__(coords, data)
        
        self.coords = coords
        self.data = data
        
    
    def prepare_data_random(self, test_data_size):
        
        """
        applying random sampling for each ensemble,
        Args: test_data_size, e.g. 0.95% 
        """
        
        
        print(type(self.coords))
        print(type(self.data))
        
        
        u_trains = []
        x_trains = []
        
        
        (
            x_train,
            u_train,
            x_test,
            u_test,
        ) = self.train_test_split(self.coords, self.data, test_data_size)
        
        print(f"Type of X: {x_train.dtype}")
        print(f"Type of Y: {u_train.dtype}")

        
       

        batch_size_1 = x_train.shape[0]

        train_loader = self.data_loader(x_train, u_train, 10000)
        test_loader = self.data_loader(x_test, u_test, 10000)
       
        

        train_test_loaders = [
            train_loader,
            test_loader,
       
        ]
        
        
        
        return x_train, u_train, train_test_loaders

    

    def train_test_split(self, x, data, test_data_size):
        

        x_train, x_test, data_train, data_test = train_test_split(
            x, data, test_size=test_data_size, random_state=42
        )
        
        x_train = np.array(x_train, dtype=np.float32)
        data_train = np.array(data_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        data_test = np.array(data_test, dtype=np.float32)
        

        return x_train, data_train, x_test, data_test
    
     
    
    def data_loader(self, X, Y, batch_size):
        X = torch.tensor(X, requires_grad=True).float().to(device)
        Y = torch.tensor(Y).float().to(device)

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=True
        )

        return train_loader


#################################################
#################################################


import torch
import numpy as np
from sklearn.model_selection import train_test_split

class LSTMDataPreparation(DataPreparation):
    def __init__(self, coords, data, noise_level=None, seq_length=10):
        super().__init__(coords, data)
        self.coords = coords
        self.data = data
        self.noise_level = noise_level
        self.seq_length = seq_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data_random(self, test_data_size):
        """
        Apply random sampling and sequence creation for LSTMs.
        Args: test_data_size, e.g. 0.2
        """
        x_train, u_train, x_test, u_test = self.train_test_split(self.coords, self.data, test_data_size)

        # Create sequences
        x_train_seq, u_train_seq = self.create_sequences(x_train, u_train, self.seq_length)
        x_test_seq, u_test_seq = self.create_sequences(x_test, u_test, self.seq_length)

        # Reshape sequences for LSTM (batch_size, seq_length, num_features)
        x_train_seq = x_train_seq.reshape(-1, self.seq_length, x_train_seq.shape[2])
        x_test_seq = x_test_seq.reshape(-1, self.seq_length, x_test_seq.shape[2])

        train_loader = self.data_loader(x_train_seq, u_train_seq, batch_size=2500)
        test_loader = self.data_loader(x_test_seq, u_test_seq, batch_size=2500, shuffle=False)

        return x_train_seq, u_train_seq, train_loader, test_loader

    def train_test_split(self, x, data, test_data_size):
        x_train, x_test, data_train, data_test = train_test_split(
            x, data, test_size=test_data_size, shuffle=False
        )
        x_train = np.array(x_train, dtype=np.float32)
        data_train = np.array(data_train, dtype=np.float32)
        x_test = np.array(x_test, dtype=np.float32)
        data_test = np.array(data_test, dtype=np.float32)
        return x_train, data_train, x_test, data_test

    def create_sequences(self, data, target, seq_length):
        sequences = []
        targets = []
        num_features = data.shape[1]  # Number of features (columns) in data

        for i in range(len(data) - seq_length):
            seq = data[i:i + seq_length]
            label = target[i + seq_length]  # Adjusted to align with the next time step
            sequences.append(seq)
            targets.append(label)

        sequences = np.array(sequences, dtype=np.float32)
        targets = np.array(targets, dtype=np.float32)

        return torch.tensor(sequences), torch.tensor(targets)

    def data_loader(self, X, Y, batch_size, shuffle=True):
        X = X.to(self.device)
        Y = Y.to(self.device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle
        )

################################################
################################################


import torch
import numpy as np
from sklearn.model_selection import train_test_split

class CNNBiLSTMDataPreparation(DataPreparation):
    def __init__(self, coords, data, noise_level=None, seq_length=10, horizon=1, stride=1):
        super().__init__(coords, data)
        self.coords = coords
        self.data = data
        self.noise_level = noise_level
        self.seq_length = seq_length
        self.horizon = horizon
        self.stride = stride
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def prepare_data_random(self, test_data_size, batch_size=2500):
        # time-ordered split (like your LSTMDataPreparation)
        x_train, u_train, x_test, u_test = self.train_test_split(self.coords, self.data, test_data_size)

        # build windows (t..t+L-1 -> y at t+L for h=1; consecutive steps for h>1)
        x_train_seq, u_train_seq = self.create_sequences(x_train, u_train, self.seq_length, self.horizon, self.stride)
        x_test_seq,  u_test_seq  = self.create_sequences(x_test,  u_test,  self.seq_length, self.horizon, self.stride)

        # reshape to (N, seq, F) — exactly what your CNN_BiLSTM.forward expects
        x_train_seq = x_train_seq.reshape(-1, self.seq_length, x_train_seq.shape[2])
        x_test_seq  = x_test_seq.reshape(-1, self.seq_length, x_test_seq.shape[2])

        train_loader = self.data_loader(x_train_seq, u_train_seq, batch_size=2500)
        test_loader = self.data_loader(x_test_seq, u_test_seq, batch_size=2500, shuffle=False)

        return x_train_seq, u_train_seq, train_loader, test_loader

    def train_test_split(self, x, data, test_data_size):
        x_train, x_test, data_train, data_test = train_test_split(
            x, data, test_size=test_data_size, shuffle=False
        )
        x_train = np.asarray(x_train, dtype=np.float32)
        data_train = np.asarray(data_train, dtype=np.float32)
        x_test  = np.asarray(x_test,  dtype=np.float32)
        data_test= np.asarray(data_test,dtype=np.float32)
        return x_train, data_train, x_test, data_test

        # horizon=3, predict 3 timesteps ahead (multi-step forecasting)
    def create_sequences(self, data, target, seq_length, horizon=1, stride=1):
        sequences = []
        targets = []
        T, F = data.shape
        D = target.shape[1] if target.ndim == 2 else 1 # D is the number of target variables, D = 1

        last_start = T - (seq_length + horizon)
        if last_start < 0:
            raise ValueError(f"Not enough timesteps T={T} for seq_length={seq_length} and horizon={horizon}.")

        for start in range(0, last_start + 1, stride):
            end = start + seq_length
            x_win = data[start:end]                 # (seq_length, F)
            y_win = target[end:end + horizon]       # (horizon, D)
            y_tar = y_win[0] if horizon == 1 else y_win.reshape(-1)  # (D,) or (horizon*D,)
            sequences.append(x_win)
            targets.append(y_tar)

        sequences = np.stack(sequences).astype(np.float32)
        targets   = np.stack(targets).astype(np.float32)

        return torch.tensor(sequences), torch.tensor(targets)


    def data_loader(self, X, Y, batch_size, shuffle=True):
        X = X.to(self.device)
        Y = Y.to(self.device)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X, Y), batch_size=batch_size, shuffle=shuffle)














  