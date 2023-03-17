from datetime import datetime
from copy import deepcopy
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

class MapBYOL(nn.Module):
    """
    The BYOL model consists on 2 networks: an online and a target network.
    Traditionally, we have an input image x, which is then transformed into positive pairs v and v'.
    We train the online network, which consists of:
        - an encoder (which converts v to a vector representation y = f(v))
        - a projector (which projects y into latent space representation z = g(y))
        - a predictor (which predicts the latent representation for v' from v_sim = q(z))
    The latent representation for v' is obtained by applying the target network to v'.
    However, we don't directly train this target network; 
    instead, we use an EXPONENTIAL MOVING AVERAGE of the weights for the online network.
    The loss for this involves applying a cosine similarity (which is L2 normalised) between 
    v_sim and the latent representation for v'. Since the model isn't symmetric, 
    we compute the same loss, but this time passing v' through the online network, 
    and v through the target network. We then add these 2 losses to obtain the BYOL loss.
    Critically, we only compute gradients with respect to the parameters of the online network
    """

    def __init__(self, encoder, encoder_parameters, projector_parameters, predictor_parameters, ema_tau):
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
        ema_tau: a float (between 0 and 1). The constant used to compute the exponential moving 
                 average when deriving the target network parameters.
        """
        super(MapBYOL, self).__init__()

        self.kwargs = {"encoder" : encoder,
                       "encoder_parameters" : encoder_parameters,
                       "projector_parameters" : projector_parameters,
                       "predictor_parameters" : predictor_parameters,
                       "ema_tau" : ema_tau}

        # model constants
        self.MAX_PIXEL_VALUE = 255
        self.RESNET_DIM = 224

        self.ema_tau = ema_tau

        self.use_resnet = encoder_parameters["use_resnet"]

        # define networks
        self.online_network = EncoderProjectorNN(encoder=encoder,
                                                 projector=MLP(**projector_parameters),
                                                 encoder_layer_idx=encoder_parameters["encoder_layer_idx"])

        self.target_network = deepcopy(self.online_network)

        self.online_predictor = MLP(**predictor_parameters)

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
    def update_target_network(self):
        """
        Updates the target network parameters by using an exponential moving average 
        of the online network parameters.
        """
        for online_params, target_params in zip(self.online_network.parameters(), self.target_network.parameters()):
            target_params.data = self.ema_tau * target_params.data + (1 - self.ema_tau) * online_params.data

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
        :param img: a torch tensor, with 3 colour channels (this must be stored in the last dimensions),
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

        return self.online_network.encode(forward_x)

    def compile_optimiser(self, **kwargs):
        """
        Sets the optimiser parameters.
        """
        self.optimiser = optim.Adam(self.parameters(), **kwargs)

    def byol_loss(self, x_1, x_2):
        """
        The BYOL loss, as stated in the BYOL paper (this applies to positive-pair batches x_1, x_2).
        Firstly, normalises the embeddings to unit vectors.
        Then, computes the dot product between positive-pair embeddings.
        The factors of 2 could be removed, but are maintained for consistency with the paper.
        """
        norm_x_1 = F.normalize(x_1, dim=-1, p=2)
        norm_x_2 = F.normalize(x_2, dim=-1, p=2)

        return 2 - 2 * (norm_x_1 * norm_x_2).sum(dim=-1)

    def get_loss(self, x_1, x_2):
        """
        Computes the loss given positive-pair batches (x_1, x_2)
        """

        # compute online encodings
        online_projection_1 = self.online_network(x_1)
        online_projection_2 = self.online_network(x_2)

        # compute target encodings
        with torch.no_grad():
            target_projection_1 = self.target_network(x_1)
            target_projection_2 = self.target_network(x_2)

        # predict target encodings from online encodings
        online_prediction_1 = self.online_predictor(online_projection_1)
        online_prediction_2 = self.online_predictor(online_projection_2)

        # compute the loss between online-predicted encodings, and target encodings
        loss_1 = self.byol_loss(online_prediction_1, target_projection_2.detach())
        loss_2 = self.byol_loss(online_prediction_2, target_projection_1.detach())

        # average the loss over the batch
        return (loss_1 + loss_2).mean()

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

            model_params_dir = os.path.join(checkpoint_dir, "byol_checkpoint.pt")
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
            val_losses.append(self.get_loss(x_1, x_2).cpu())

        self.train()

        return np.mean(val_losses)

    def train_model(self, train_loader, validation_loader, epochs, checkpoint_dir = None, logs_per_epoch = 100, evaluations_per_epoch = 100):
        """
        Trains the network.
        ---------------------------------------------------------------------------------------------------
        :param train_loader: a PyTorch DataLoader, containing the training data.
        :param validation_loader: a PyTorch DataLoader, containing the validation data.
        :param epochs: an int, the number of epochs for training.
        :param checkpoint_dir: a string, the directory to which to write the checkpoints.
        :param logs_per_epoch: an int. Every logs_per_epoch batches, the performance of the network
                               is logged. By default logging is performed every 100 batches.
        
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

        for epoch in range(epochs):
            batch_losses = []
            validation_losses = []
            avg_batch_losses_20 = []
            logging.info(f"Starting Epoch: {epoch + 1}")

            for batch, (x_1, x_2) in enumerate(train_loader):
                # x_1 and x_2 are tensors containing patches, 
                # such that x_1[i] and x_2[i] are patches for the same area
                self.optimiser.zero_grad()

                x_1, x_2 = transform_inputs(x_1.to(self.device)), transform_inputs(x_2.to(self.device))

                loss = self.get_loss(x_1, x_2)

                batch_losses.append(loss.cpu().detach())

                loss.backward()
                self.optimiser.step()
                self.update_target_network()

                if batch % (len(train_loader) // logs_per_epoch + 1) == 0 and batch != 0:
                    with torch.no_grad():
                        avg_loss = np.mean(batch_losses[-20:])
                        avg_batch_losses_20.append(avg_loss)
                        logging.info(
                            f"Epoch {epoch + 1}: [{batch + 1}/{len(train_loader)}] ---- BYOL Training Loss = {avg_loss}")

                if batch % (len(train_loader) // evaluations_per_epoch + 1) == 0:
                    validation_loss = self.get_validation_loss(validation_loader)
                    validation_losses.append(validation_loss)

                    if validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        best_model_state_dict = self.state_dict()
                        logging.info(f"Epoch {epoch + 1} ---- New Best BYOL Validation Loss = {validation_loss}")
                    else:
                        logging.info(f"Epoch {epoch + 1} ---- BYOL Validation Loss = {validation_loss}")

                    self.update_checkpoint(checkpoint_dir=checkpoint_dir,
                                           batch_losses=batch_losses,
                                           validation_losses=validation_losses,
                                           epoch=epoch,
                                           batch=batch,
                                           model_state_dict=self.state_dict(),
                                           best_model_state_dict=best_model_state_dict,
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
                                       best_model_state_dict=best_model_state_dict,
                                       optimiser_state_dict=self.optimiser.state_dict,
                                       loss=loss.cpu().detach(),
                                       avg_batch_losses_20=avg_batch_losses_20,
                                       run_end=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        return self.checkpoint
