from datetime import datetime
import os
import pickle as pk
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format="%(asctime)s %(levelname)-4s %(message)s",
                    level=logging.INFO,
                    datefmt="%d-%m-%Y %H:%M:%S")

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

from mlp import MLP
from encoder_projector_nn import EncoderProjectorNN

class MapSIMCLR(nn.Module):
    def __init__(self, encoder, encoder_parameters, projector_parameters, tau):
        """
        encoder: a nn.Module, containing an encoder network.
        encoder_parameters: a dict with 2 keys:
                            - encoder_layer_idx: an int, corresponding to the index of the layer of the encoder
                                                   which is actually used for encoding.
                                                   For instance, in BYOL, if using a ResNet as an encoder,
                                                   the paper uses the output of the last average pooliing layer,
                                                   which is the penultimate layer of the ResNet
                                                   (corresponding to encoder_layer_idx = -2).
                            - use_resnet: a boolean, corresponding to whether the encoder is a ResNet or a CNN.
        projector_parameters: a dict, containing the parameters to initialise an MLP to act as
                              a projector network.
        projector_parameters: a dict, containing the parameters to initialise an MLP to act as
                              a predictor network.
        tau: temperature parameter for NT-XENT loss
        """
        super(MapSIMCLR, self).__init__()

        self.kwargs = {"encoder": encoder,
                       "encoder_parameters": encoder_parameters,
                       "projector_parameters": projector_parameters,
                       "tau": tau}
        
        # model constants
        self.MAX_PIXEL_VALUE = 255
        self.RESNET_DIM = 224

        self.tau = tau

        self.use_resnet = encoder_parameters["use_resnet"]
        
        # define the model
        self.model = EncoderProjectorNN(encoder = encoder,
                                        projector = MLP(**projector_parameters),
                                        encoder_layer_idx = encoder_parameters["encoder_layer_idx"])
        
        # define optimiser
        self.optimiser = None

        # get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # create checkpoint
        self.checkpoint = {"epoch": 0,
                           "batch": 0,
                           "model_state_dict": self.state_dict(),
                           "best_model_state_dict": None,
                           "optimiser_state_dict": None,
                           "loss": 0,
                           "avg_batch_losses_20": [],
                           "batch_losses": [],
                           "validation_losses ": [],
                           "run_start": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                           "run_end": None,
                           "model_kwargs": self.kwargs}

    @torch.no_grad()
    def to_tensor(self, img):
        if len(img.shape) == 3:
            norm_img = torch.moveaxis(img, -1, 0)
        else:
            norm_img = torch.moveaxis(img, -1, 1)

        if torch.max(img) > 1:
            norm_img = norm_img / self.MAX_PIXEL_VALUE

        return norm_img

    @torch.no_grad()
    def img_to_resnet(self, img, dim=None):
        """
        Convert image into the desired format for ResNet.
        The image must have width and height of at least self.RESNET_DIM, with RGB values between 0 and 1.
        Moreover, it must be normalised, by using a mean of [0.485, 0.456, 0.406] and a standard deviation
        of [0.229, 0.224, 0.225]
        ---------------------------------------------------------------------------------------------------
        :param img: a numpy nd.array, with 3 colour channels (this must be stored in the last dimensions),
        which has to be fed to ResNet
        :param dim: the desired dimension of the image (if we want to resize img before feeding it to
                    ResNet). This should be at least self.RESTNET_DIM.
        ---------------------------------------------------------------------------------------------------
        :return a Tensor, with the first dimension corresponding to the RGB channels, and normalised to be
                used by ResNet.
        """

        # put the colour channel in front and normalise into range [0,1]
        norm_img = self.to_tensor(img)

        # resize
        if dim is not None:
            assert dim >= self.RESNET_DIM, f"Provided dimension {dim} is less than the required for RESNET ({self.RESNET_DIM})"
            norm_img = T.Resize(dim)(norm_img)
        else:
            norm_img = T.Resize(self.RESNET_DIM)(norm_img)

        # normalise mean and variance
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])

        return T.Normalize(mean=mean, std=std)(norm_img)

    def forward(self, x):
        """
        Returns the encoding (without projection) of the input, corresponding to the online network.
        """
        if self.use_resnet:
            forward_x = self.img_to_resnet(x)
        else:
            forward_x = self.to_tensor(x)

        return self.model.encode(forward_x)

    def compile_optimiser(self, **kwargs):
        """
        Sets the optimiser parameters.
        """
        self.optimiser = optim.Adam(self.parameters(), **kwargs)
    
    def contrastive_loss(self, z_batch):
        """
        Computes the contrastive loss (NT-XENT) for a mini-batch of augmented samples.
        --------------------------------------------------------------------------------------------------------
        z_batch: a (N,K) Tensor, with rows as embedding vectors. 
                 We expect that z_batch[2k] and z_batch[2k+1], 0 <= k < N, correspond to a positive sample pair
        --------------------------------------------------------------------------------------------------------
        return: a float, corresponding to the total loss for the mini-batch z_batch
        """
        N = len(z_batch)

        # normalise to have unit length rows
        norm_z_batch = F.normalize(z_batch)

        # compute similarity & apply factor of tau
        sim_batch = (norm_z_batch @ norm_z_batch.T)/self.tau

        # fill the diagonal with -1000, to make sure it is never considered in the cross entropy computations
        sim_batch.fill_diagonal_(-1000)

        # generate labels
        # z_batch[2k] should be similar to z_batch[2k+1] (since these will be the positive pair)
        # hence, labels should have the form [1,0,3,2,...,N,N-1]
        labels = torch.Tensor([k+1 if k%2 == 0 else k-1 for k in range(0,N)]).long().to(self.device)

        # return the NT-XENT loss
        return 1/N * F.cross_entropy(sim_batch, labels, reduction = "sum")

    def get_loss(self, x_1, x_2):
        """
        Computes the loss given positive-pair batches (x_1, x_2)
        """

        z_1 = self.model(x_1)
        z_2 = self.model(x_2)

        z_batch = torch.stack((z_1, z_2), dim=1).view(-1, z_1.shape[1])

        return self.contrastive_loss(z_batch)

    @torch.no_grad()
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

            model_params_dir = os.path.join(checkpoint_dir, "simclr_checkpoint.pt")
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
    def get_validation_loss(self, validation_loader):
        """
        Computes the average validation loss of the model.
        """
        val_losses = []

        self.eval()

        if self.use_resnet:
            transform_inputs = self.img_to_resnet
        else:
            transform_inputs = self.to_tensor

        for x_1, x_2 in validation_loader:
            x_1, x_2 = transform_inputs(x_1.to(self.device)), transform_inputs(x_2.to(self.device))
            loss = self.get_loss(x_1, x_2)

            val_losses.append(loss.cpu())

        self.train()

        return np.mean(val_losses)
    
    def train_model(self,
                    train_loader,
                    validation_loader,
                    epochs,
                    checkpoint_dir=None,
                    logs_per_epoch=100,
                    evaluations_per_epoch=100,
                    patience_prop=0.25):
        """
        Trains the network.
        """
        if self.optimiser is None:
            logging.warning("Can't train if the optimiser hasn't been set. Please run model.compile_optimiser first.")
            return None

        self.to(self.device)

        if self.use_resnet:
            transform_inputs = self.img_to_resnet
        else:
            transform_inputs = self.to_tensor

        best_validation_loss = float("inf")
        best_model_state_dict = None

        if 0 <= patience_prop <= 1:
            patience = int(patience_prop * evaluations_per_epoch)
        else:
            patience = float("inf")

        logging.info(f"Early stopping patience set to {patience}")

        for epoch in range(epochs):
            batch_losses = []
            validation_losses = []
            avg_batch_losses_20 = []
            n_runs_no_improvement = 0
            logging.info(f"Starting Epoch: {epoch + 1}")

            for batch, (x_1, x_2) in enumerate(train_loader):
                # x_1 and x_2 are tensors containing patches, 
                # such that x_1[i] and x_2[i] are patches for the same area
                
                self.optimiser.zero_grad()

                x_1, x_2 = transform_inputs(x_1.to(self.device)), transform_inputs(x_2.to(self.device))
                loss = self.get_loss(x_1, x_2)

                batch_losses.append(loss.detach().cpu())
                
                loss.backward()
                self.optimiser.step()

                if batch % (len(train_loader) // logs_per_epoch) == 0 and batch != 0:
                    with torch.no_grad():
                        avg_loss = np.mean(batch_losses[-20:])
                        avg_batch_losses_20.append(avg_loss)
                        logging.info(f"Epoch {epoch + 1}: [{batch + 1}/{len(train_loader)}] ---- NT-XENT Training Loss = {avg_loss}")

                if batch % (len(train_loader) // evaluations_per_epoch) == 0:
                    validation_loss = self.get_validation_loss(validation_loader)
                    validation_losses.append(validation_loss)

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_model_state_dict = self.state_dict()
                        logging.info(f"Epoch {epoch + 1}: [{batch + 1}/{len(train_loader)}] ---- New Best NT-XENT Validation Loss = {validation_loss}")
                        n_runs_no_improvement = 0
                    else:
                        logging.info(f"Epoch {epoch + 1}: [{batch + 1}/{len(train_loader)}] ---- NT-XENT Validation Loss = {validation_loss}")
                        n_runs_no_improvement += 1

                    self.update_checkpoint(checkpoint_dir=checkpoint_dir,
                                           batch_losses=batch_losses,
                                           validation_losses=validation_losses,
                                           epoch=epoch,
                                           batch=batch,
                                           model_state_dict=self.state_dict(),
                                           best_model_state_dict=best_model_state_dict,
                                           optimiser_state_dict=self.optimiser.state_dict,
                                           loss=loss.detach().cpu(),
                                           avg_batch_losses_20=avg_batch_losses_20,
                                           run_end=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

                    if (n_runs_no_improvement >= patience and epoch > 4) \
                            or (n_runs_no_improvement >= evaluations_per_epoch and patience < float("inf")):
                        logging.info(f"Stopping training, at epoch={epoch + 1}, batch={batch + 1} "
                                     f"after no validation improvement in {patience} consecutive evaluations "
                                     f"after 5 epochs, or no improvement during a whole epoch.\n"
                                     f"Best Validation: {best_validation_loss}")

                        return self.checkpoint

            with torch.no_grad():
                self.update_checkpoint(checkpoint_dir=checkpoint_dir,
                                       batch_losses=batch_losses,
                                       validation_losses=validation_losses,
                                       epoch=epoch,
                                       batch=batch,
                                       model_state_dict=self.state_dict(),
                                       best_model_state_dict=best_model_state_dict,
                                       optimiser_state_dict=self.optimiser.state_dict,
                                       loss=loss.detach().cpu(),
                                       avg_batch_losses_20=avg_batch_losses_20,
                                       run_end=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        return self.checkpoint
