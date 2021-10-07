"""
AudioSpectDataset.py
Author: Saurabh Biswas
Institution: University of Osnabrueck

Handles input data to the neural network

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

    def __init__(self, spectrograms, labels, transforms=None):  # if directly created then ensure same len for spect and label lists
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

    def get_all_labels(self):
        return self.labels

    def get_datasets(inputfile, label_dict=None, train_ratio=0.80, transforms=None):
        data = pd.read_csv(inputfile)
        train_spects, test_spects, train_labels, test_labels = \
            train_test_split(data['spectrogram'], data['label'],
                             train_size=train_ratio, random_state=3, shuffle=True, stratify=data['label'])
        print("****Training set****\n",train_labels.value_counts())
        print("****Validation set****\n",test_labels.value_counts())
        train_labels = train_labels.map(label_dict).astype(int)
        test_labels = test_labels.map(label_dict).astype(int)

        return AudioSpectDataset(train_spects.to_list(), train_labels.to_list(), transforms), \
               AudioSpectDataset(test_spects.to_list(), test_labels.to_list(), transforms)
