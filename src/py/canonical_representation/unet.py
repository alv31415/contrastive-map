from datetime import datetime
import os
import pickle as pk
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%d-%m-%Y %H:%M:%S')

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Downsampler(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsampler, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.mpconv(x)


class Upsampler(nn.Module):
    def __init__(self, in_channels, out_channels, use_contrastive_model):
        super(Upsampler, self).__init__()

        self.use_contrastive_model = use_contrastive_model

        if use_contrastive_model:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        else:
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x_1, x_2):
        x_1 = self.up(x_1)
        x = x_1 if self.use_contrastive_model else torch.cat([x_2, x_1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, contrastive_model=None):
        super(UNet, self).__init__()
        self.contrastive_model = contrastive_model

        self.MAX_PIXEL_VALUE = 255
        self.RESNET_DIM = 224


        self.kwargs = {"contrastive_model": contrastive_model}
        self.use_contrastive_model = False

        # freeze contrastive model, we don't want to train it, just encode images
        if contrastive_model is not None:
            self.use_contrastive_model = True
            self.contrastive_img_width = 8
            self.contrastive_hidden_dim = 512
            self.contrastive_bottleneck = nn.Linear(in_features=self.contrastive_hidden_dim,
                                                    out_features=self.contrastive_img_width
                                                                 * self.contrastive_img_width
                                                                 * self.contrastive_hidden_dim)

            for param in contrastive_model.parameters():
                param.requires_grad = False

        self.input = None if self.use_contrastive_model else DoubleConv(in_channels=3, out_channels=64)
        self.downsampler1 = None if self.use_contrastive_model else Downsampler(in_channels=64, out_channels=128)
        self.downsampler2 = None if self.use_contrastive_model else Downsampler(in_channels=128, out_channels=256)
        self.downsampler3 = None if self.use_contrastive_model else Downsampler(in_channels=256, out_channels=512)
        self.downsampler4 = None if self.use_contrastive_model else Downsampler(in_channels=512, out_channels=512)

        self.upsampler1 = Upsampler(in_channels=512, out_channels=256, use_contrastive_model=self.use_contrastive_model)
        self.upsampler2 = Upsampler(in_channels=256, out_channels=128, use_contrastive_model=self.use_contrastive_model)
        self.upsampler3 = Upsampler(in_channels=128, out_channels=64, use_contrastive_model=self.use_contrastive_model)
        self.upsampler4 = Upsampler(in_channels=64, out_channels=32, use_contrastive_model=self.use_contrastive_model)
        self.output = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=3, kernel_size=1),
                                    nn.Sigmoid())

        self.criterion = None
        self.optimiser = None

        self.test_imgs = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # create checkpoint
        self.checkpoint = {"epoch": 0,
                           "batch": 0,
                           "model_state_dict": self.state_dict(),
                           "optimiser_state_dict": None,
                           "loss": 0,
                           "avg_batch_losses_20": [],
                           "batch_losses": [],
                           "validation_losses ": [],
                           "run_start": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                           "run_end": None,
                           "model_kwargs": self.kwargs}

    def forward(self, x):
        if self.use_contrastive_model:
            x_down = self.contrastive_model(x)
            x_down = self.contrastive_bottleneck(x_down)
            x_down = x_down.view(-1,
                                 self.contrastive_hidden_dim,
                                 self.contrastive_img_width,
                                 self.contrastive_img_width)
            x_up = self.upsampler1(x_down, None)
            x_up = self.upsampler2(x_up, None)
            x_up = self.upsampler3(x_up, None)
            x_up = self.upsampler4(x_up, None)
            x_up = self.output(x_up)
        else:
            x_down_1 = self.input(x)
            x_down_2 = self.down1(x_down_1)
            x_down_3 = self.down2(x_down_2)
            x_down_4 = self.down3(x_down_3)
            x_down_5 = self.down4(x_down_4)
            x_up = self.up1(x_down_5, x_down_4)
            x_up = self.up2(x_up, x_down_3)
            x_up = self.up3(x_up, x_down_2)
            x_up = self.up4(x_up, x_down_1)
            x_up = self.output(x_up)

        # move colour channel back to the last dimension
        return torch.movedim(x_up, 1, -1)

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, model, model_kwargs=None, use_resnet=True):

        logging.info(f"Loading model from checkpoint: {checkpoint_dir}")
        checkpoint = torch.load(checkpoint_dir, map_location=torch.device('cpu'))

        try:
            if "model_kwargs" in checkpoint:
                model_kwargs = checkpoint["model_kwargs"]
                if "encoder_layer_idx" in model_kwargs:
                    model_kwargs["encoder_parameters"] = {"encoder_layer_idx": model_kwargs["encoder_layer_idx"],
                                                          "use_resnet": use_resnet}
                    model_kwargs.pop("encoder_layer_idx")
                cl_model = model(**model_kwargs)
            else:
                cl_model = model(**model_kwargs)
        except:
            logging.info("No model was loaded. Invalid arguments provided")
            logging.info(f"Kwargs provided: {model_kwargs}")
            logging.info(f"Kwargs found in checkpoint: {checkpoint['model_kwargs']}")

            return None

        cl_model.load_state_dict(state_dict=checkpoint["model_state_dict"])

        logging.info("Model loaded and set to evaluation mode")
        cl_model.eval()

        return cls(cl_model)

    def compile_model(self, test_imgs = None, loss_str = "MSE", optimiser = optim.Adam, **optim_kwargs):

        if test_imgs is None:
            self.test_imgs = (torch.rand(1,128,128,3) * self.MAX_PIXEL_VALUE).int()
        else:
            self.test_imgs = test_imgs

        if loss_str == "MSE":
            self.criterion = nn.MSELoss()
        elif loss_str == "L1":
            self.criterion = nn.L1Loss()
        else:
            logging.info(f"Provided loss string '{loss_str}' is invalid (must be one of 'MSE' or 'L1'. Defaulting to MSE loss.")


        self.optimiser = optimiser(self.parameters(), **optim_kwargs)

    def norm_img(self, img):
        if torch.max(img) > 1:
            return img / self.MAX_PIXEL_VALUE

        return img

    def get_loss(self, reconstruction, target):
        return self.criterion(reconstruction, target)

    @torch.no_grad()
    def evaluate(self, evaluation_loader, validation=True):

        eval_losses = []

        self.eval()

        for original, target in evaluation_loader:
            original, target = original.to(self.device), self.norm_img(target).to(self.device)
            reconstruction = self(original)
            eval_losses.append(self.get_loss(reconstruction = reconstruction, target = target))

            del original
            del target
            torch.cuda.empty_cache()

        if validation:
            self.train()
            return np.mean(eval_losses)

        return eval_losses

    def update_checkpoint(self, checkpoint_dir, batch_losses, validation_losses, **checkpoint_data):
        """
        Updates the checkpoint dictionary.
        """
        for k, v in checkpoint_data.items():
            if k in self.checkpoint:
                self.checkpoint[k] = v

        if checkpoint_dir is not None:

            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            model_params_dir = os.path.join(checkpoint_dir, "unet_checkpoint.pt")
            torch.save(self.checkpoint, model_params_dir)

            batch_loss_train_dir = os.path.join(checkpoint_dir,
                                                f"batch_loss_logs_t{checkpoint_data.get('epoch', 0)}.pk")
            with open(batch_loss_train_dir, "wb") as f:
                pk.dump(batch_losses, f)

            batch_loss_validation_dir = os.path.join(checkpoint_dir,
                                                f"batch_loss_logs_v{checkpoint_data.get('epoch', 0)}.pk")
            with open(batch_loss_validation_dir, "wb") as f:
                pk.dump(validation_losses, f)

    @torch.no_grad()
    def save_reconstructions(self, checkpoint_dir, epoch, batch):
        reconstructions = self(self.test_imgs)

        for i, reconstruction in enumerate(reconstructions):
            with open(os.path.join(checkpoint_dir, reconstructions, f"{i}_b{batch}_e{epoch}.png"), "wb") as f:
                Image.fromarray(np.array(reconstruction)).save(f)

    def train_model(self, train_loader, validation_loader, epochs, checkpoint_dir=None, batch_log_rate=100,
                    save_reconstruction_interval=100):
        """
        Trains the network.
        """

        if self.criterion is None or self.optimiser is None:
            logging.warning("Can't train if the optimiser or loss haven't been set. Please run model.compile_model first.")
            return None

        self.to(self.device)

        for epoch in range(epochs):
            batch_losses = []
            validation_losses = []
            avg_batch_losses_20 = []
            logging.info(f"Starting Epoch: {epoch + 1}")
            for batch, (original, target) in enumerate(train_loader):

                self.optimiser.zero_grad()

                original, target = original.to(self.device), self.norm_img(target).to(self.device)
                reconstruction = self(original)

                loss = self.get_loss(reconstruction = reconstruction, target = target)

                batch_losses.append(loss.cpu().detach())

                loss.backward()
                self.optimiser.step()

                if batch % save_reconstruction_interval == 0:
                    self.save_reconstructions(checkpoint_dir = checkpoint_dir, epoch = epoch, batch = batch)

                if batch % (len(train_loader) // batch_log_rate + 1) == 0 and batch != 0:
                    with torch.no_grad():
                        avg_loss = np.mean(batch_losses[-20:])
                        avg_batch_losses_20.append(avg_loss)
                        logging.info(
                            f"Epoch {epoch + 1}: [{batch + 1}/{len(train_loader)}] ---- NT-XENT Training Loss = {avg_loss}")

                        if batch % (len(train_loader) // (batch_log_rate // 4) + 1) == 0:
                            validation_loss = self.get_validation_loss(validation_loader)
                            validation_losses.append(validation_loss)
                            logging.info(
                                f"Epoch {epoch + 1}: [{batch + 1}/{len(train_loader)}] ---- NT-XENT Validation Loss = {validation_loss}")

                        self.update_checkpoint(checkpoint_dir=checkpoint_dir,
                                               batch_losses=batch_losses,
                                               validation_losses=validation_losses,
                                               epoch=epoch,
                                               batch=batch,
                                               model_state_dict=self.state_dict(),
                                               optimiser_state_dict=self.optimiser.state_dict,
                                               loss=loss.cpu().detach(),
                                               avg_batch_losses_20=avg_batch_losses_20,
                                               run_end=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            with torch.no_grad():
                self.update_checkpoint(checkpoint_dir=checkpoint_dir,
                                       batch_losses=batch_losses,
                                       validation_losses=validation_losses,
                                       epoch=epochs,
                                       batch=len(train_loader),
                                       model_state_dict=self.state_dict(),
                                       optimiser_state_dict=self.optimiser.state_dict,
                                       loss=loss.cpu().detach(),
                                       avg_batch_losses_20=avg_batch_losses_20,
                                       run_end=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        return self.checkpoint








