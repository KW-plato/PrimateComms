"""
Module: ChimpModel.py
Author: Saurabh Biswas
Institution: University of Osnabrueck
Created on: 20.02.2021

handles input data to the neural network

"""
from torch.utils import data
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

class AudioSpectDataset(data.Dataset):

    """
    Contains the dataset for training and validation:
    1. Not created directly, but by calling the function that reads the main input file and spilts it into train & test
    2. Stores locations of the spectrograms and corresponding labels
    3. Reads the spectrogram from disk, applies transforms if any and returns a dict with the spectrogram and label
    """

    def __init__(self, spectrograms, labels, transforms=None):  # calling program must check len of spect and label lists are same
        self.spectrograms = spectrograms
        self.labels = labels
        self.transforms = transforms
        self.length = len(labels)

    def __getitem__(self, index):
        with open(self.spectrograms[index], 'rb') as fread:
            spect = torch.from_numpy(np.load(fread)).float().unsqueeze(dim=0)
            if self.transforms:
                spect = self.transforms(spect)
        return (
            {
            'spectrogram': spect,
            'label': self.labels[index]
            }
        )

    def __len__(self):
        return self.length

    def get_datasets(inputfilelocation, label_dict=None, train_ratio=0.90, transforms=None):
        data = pd.read_csv(inputfilelocation)
        data['label'] = data.label.map(label_dict).astype(int)
        file_locations = data.iloc[:, 0].tolist()
        labels = data.iloc[:, 1].tolist()
        train_spects, test_spects, train_labels, test_labels = \
            train_test_split(file_locations, labels, train_size=train_ratio, random_state=3, shuffle=True)

        return AudioSpectDataset(train_spects, train_labels, transforms), \
               AudioSpectDataset(test_spects, test_labels, transforms)
