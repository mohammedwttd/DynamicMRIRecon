# Adapted from Nicola Dinsdale 2020
# 2D version of STAMP UNet for MRI reconstruction
########################################################################################################################
from collections import OrderedDict

import torch
import torch.nn as nn
########################################################################################################################

class UNet2D(nn.Module):
    """
    2D STAMP UNet with per-layer targeted dropout.
    
    Adapted from official STAMP: https://github.com/nkdinsdale/STAMP.git
    Original was 3D, this is 2D for MRI reconstruction.
    """

    def __init__(self, drop_probs, in_channels=1, out_channels=1, init_features=32):
        super(UNet2D, self).__init__()

        self.drop_probs = drop_probs
        self.in_channels = in_channels
        self.out_channels = out_channels

        features = init_features
        self.encoder1 = UNet2D._block(self.drop_probs, 0, 1, in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = UNet2D._block(self.drop_probs, 2, 3, features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = UNet2D._block(self.drop_probs, 4, 5, features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = UNet2D._block(self.drop_probs, 6, 7, features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = UNet2D._block(self.drop_probs, 8, 9, features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = UNet2D._block(self.drop_probs, 10, 11, (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = UNet2D._block(self.drop_probs, 12, 13, (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = UNet2D._block(self.drop_probs, 14, 15, (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet2D._half_block(self.drop_probs, 16, features * 2, features, name="dec1")

        # Output conv
        self.conv = nn.Conv2d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        return self.conv(dec1)

    @staticmethod
    def _block(drop_probs, key1, key2, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                    (name + 'drop1', nn.Dropout2d(p=drop_probs[key1])),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu2", nn.ReLU(inplace=True)),
                    (name + "norm2", nn.InstanceNorm2d(num_features=features)),
                    (name + 'drop2', nn.Dropout2d(p=drop_probs[key2])),
                ]
            )
        )

    @staticmethod
    def _half_block(drop_probs, key1, in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (name + "norm1", nn.InstanceNorm2d(num_features=features)),
                    (name + 'drop1', nn.Dropout2d(p=drop_probs[key1])),
                ]
            )
        )


def update_droplayer(model, new_drop):
    """Update dropout layers with new probabilities."""
    model.encoder1.enc1drop1 = nn.Dropout2d(p=new_drop[0])
    model.encoder1.enc1drop2 = nn.Dropout2d(p=new_drop[1])
    model.encoder2.enc2drop1 = nn.Dropout2d(p=new_drop[2])
    model.encoder2.enc2drop2 = nn.Dropout2d(p=new_drop[3])
    model.encoder3.enc3drop1 = nn.Dropout2d(p=new_drop[4])
    model.encoder3.enc3drop2 = nn.Dropout2d(p=new_drop[5])
    model.encoder4.enc4drop1 = nn.Dropout2d(p=new_drop[6])
    model.encoder4.enc4drop2 = nn.Dropout2d(p=new_drop[7])
    model.bottleneck.bottleneckdrop1 = nn.Dropout2d(p=new_drop[8])
    model.bottleneck.bottleneckdrop2 = nn.Dropout2d(p=new_drop[9])
    model.decoder4.dec4drop1 = nn.Dropout2d(p=new_drop[10])
    model.decoder4.dec4drop2 = nn.Dropout2d(p=new_drop[11])
    model.decoder3.dec3drop1 = nn.Dropout2d(p=new_drop[12])
    model.decoder3.dec3drop2 = nn.Dropout2d(p=new_drop[13])
    model.decoder2.dec2drop1 = nn.Dropout2d(p=new_drop[14])
    model.decoder2.dec2drop2 = nn.Dropout2d(p=new_drop[15])
    model.decoder1.dec1drop1 = nn.Dropout2d(p=new_drop[16])
    return model


def get_default_drop_probs(b_drop=0.1):
    """Get default dropout probabilities (all at base level)."""
    base_prob = 0.05
    return {i: b_drop * base_prob for i in range(17)}


# Wrapper class for compatibility with DynamicMRIRecon
class STAMPUnet(nn.Module):
    """
    STAMP UNet wrapper for DynamicMRIRecon compatibility.
    
    Uses the official STAMP UNet2D with targeted dropout.
    """
    
    def __init__(self, in_chans=1, out_chans=1, chans=32, num_pool_layers=4,
                 drop_prob=0.0, b_drop=0.1, mode='Taylor'):
        super().__init__()
        
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.b_drop = b_drop
        self.mode = mode
        
        # Initialize dropout probabilities
        self.drop_probs = get_default_drop_probs(b_drop)
        
        # Create the UNet
        self.unet = UNet2D(
            drop_probs=self.drop_probs,
            in_channels=in_chans,
            out_channels=out_chans,
            init_features=chans
        )
    
    def forward(self, x):
        return self.unet(x)
    
    def update_dropout_probs(self, new_probs):
        """Update dropout probabilities."""
        self.drop_probs = new_probs
        update_droplayer(self.unet, new_probs)


# Alias
STAMPReconUnet = STAMPUnet

