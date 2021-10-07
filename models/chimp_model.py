"""
ChimpModel.py
Author: Saurabh Biswas
Institution: University of Osnabrueck

Defines the NN model for classifying chimpanzee vocalisation samples

Ref: Oikarinen, T., Srinivasan, K. ,Meisner, O., et al. (2019) Deep convolutional network for animal sound
classification and source attribution using dual audio recordings. The Journal of the Acoustical Society of America
145, 654. doi: https://doi.org/10.1121/1.5087827

"""

import torch.nn as nn

class ChimpCallClassifier(nn.Module):
    def __init__(self, num_labels=4, spectrogram_shape=(257, 254),dropout=0.5):
        """
        Args:
            num_labels (int): No. of labels to be predicted.
                              Default = 4
            spectrogram_shape (tuple of int): Dimensions of input spectrogram.
                                              Default = (257,254)
            dropout : dropout ratio
                      Default = 0.5

        """
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            )
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(
                in_features=64*(spectrogram_shape[0]//32)*(spectrogram_shape[1]//32),
                out_features=1024,
                bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(
                in_features=1024,
                out_features=num_labels,
                bias=False
            )
        )

    def forward(self, x):
        """
        One forward pass through the models
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)

        return x
