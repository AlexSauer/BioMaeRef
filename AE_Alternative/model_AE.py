from torch import nn
import torch
import lightning as L
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from einops import rearrange


class Encoder(nn.Module):
    def __init__(
        self,
        num_input_channels: int,
        base_channel_size: int,
        depth: int,
        image_size: int,
        act_fn: object = nn.GELU,
        double_int: int = 2,
    ):
        """Encoder.

        Args:
           num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           depth: Number of convolutional layers in each stage
           image_size: Size of the input image (assuming square images)
           act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        self.depth = depth
        c_hid = base_channel_size
        layers = []
        in_size = image_size

        for i in range(depth):
            if i % double_int == 0:
                c_hid *= 2
            layers.append(nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2))
            layers.append(nn.BatchNorm2d(c_hid))
            layers.append(act_fn())
            num_input_channels = c_hid
            in_size //= 2

        self.latent_dim = c_hid
        latent_dim = (in_size**2) * c_hid
        self.linear = nn.Linear(latent_dim, latent_dim)
        self.final_size = in_size

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        latent = self.net(x)
        latent = latent.view(-1, self.latent_dim * self.final_size**2)
        return self.linear(latent)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        depth: int,
        last_image_size: int,
        num_output_channels: int,
        act_fn: object = nn.GELU,
        double_int: int = 2,
    ):
        """Decoder.

        Args:
           latent_dim : Dimensionality of latent representation z
           base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
           depth: Number of convolutional layers in each stage
           image_size: Size of the input image (assuming square images)
           num_output_channels: Number of output channels of the image
           act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        self.depth = depth
        self.latent_dim = latent_dim
        c_hid = latent_dim

        layers = []
        self.last_image_size = last_image_size
        self.init_latent_dim = latent_dim

        for i in range(depth - 1):
            if i % double_int == 0:
                c_hid //= 2
            layers.append(nn.ConvTranspose2d(latent_dim, c_hid, kernel_size=3, padding=1, stride=2, output_padding=1))
            layers.append(nn.BatchNorm2d(c_hid))
            layers.append(act_fn())
            layers.append(nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(c_hid))
            layers.append(act_fn())
            latent_dim = c_hid
            last_image_size *= 2

        layers.append(
            nn.ConvTranspose2d(latent_dim, num_output_channels, kernel_size=3, padding=1, stride=2, output_padding=1)
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.init_latent_dim, self.last_image_size, self.last_image_size)
        return torch.sigmoid(self.net(x))


class AE_Module(L.LightningModule):
    def __init__(
        self,
        depth,
        base_channel_size,
        entropy_loss_threshold,
        classes=5,
        lr=0.00004,
        weights=[1, 1, 1, 1, 1],
        double_int=2,
    ):
        super().__init__()
        self.depth = depth
        self.lr = lr
        self.weights = weights
        self.entropy_loss_threshold = entropy_loss_threshold
        self.encoder = Encoder(
            num_input_channels=1,
            base_channel_size=base_channel_size,
            depth=depth,
            image_size=128,
            act_fn=nn.GELU,
            double_int=double_int,
        )
        self.decoder = Decoder(
            latent_dim=self.encoder.latent_dim,
            depth=depth,
            last_image_size=self.encoder.final_size,
            num_output_channels=classes,
            act_fn=nn.GELU,
            double_int=double_int,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def training_step(self, batch, batch_idx):
        mito, entropy, dist = batch
        target = mito.clone()
        x_hat = self(mito)
        target[entropy > self.entropy_loss_threshold] = -1
        loss = F.cross_entropy(
            x_hat, target.squeeze().long(), ignore_index=-1, weight=torch.tensor(self.weights).to(self.device)
        )
        self.log('train_loss', loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': StepLR(optimizer, step_size=10, gamma=0.1),  # Stepwise scheduler after 5 epochs
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [scheduler]
