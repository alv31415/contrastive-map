from datetime import datetime
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F

import wandb

from mlp import MLP
from encoder_projector_nn import EncoderProjectorNN

class MapSIMCLR(nn.Module):
    def __init__(self, encoder, encoder_layer_idx, projector_parameters, tau):
        """
        tau: temperature parameter for NT-XENT loss
        """
        super(MapSIMCLR, self).__init__()
        
        # model constants
        self.MAX_PIXEL_VALUE = 255
        self.RESNET_DIM = 224
        
        # define the model
        self.model = EncoderProjectorNN(encoder = encoder,
                                        projector = MLP(**projector_parameters),
                                        encoder_layer_idx = encoder_layer_idx)
        
        self.tau = tau
        
        # define optimiser
        self.optimiser = None

        # get the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # create checkpoint
        self.checkpoint = {"epoch" : 0,
                           "batch" : 0,
                           "model_state_dict" : self.state_dict(),
                           "optimiser_state_dict": None,
                           "loss" : 0,
                           "avg_loss_20" : 0,
                           "run_start" : datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                           "run_end" : None}
    
    def img_to_resnet(self, img, dim = None):
        """
        Convert image into the desired format for ResNet.
        The image must have width and height of at least self.RESNET_DIM, with RGB values between 0 and 1.
        Moreover, it must be normalised, by using a mean of [0.485, 0.456, 0.406] and a standard deviation of [0.229, 0.224, 0.225]
        --------------------------------------------------------------------------------------------------------------------------------
        :param img: a numpy nd.array, with 3 colour channels (this must be stored in the last dimensions), which has to be fed to ResNet
        :param dim: the desired dimension of the image (if we want to resize img before feeding it to ResNet).
                    This should be at least self.RESTNET_DIM.
        --------------------------------------------------------------------------------------------------------------------------------
        :return a Tensor, with the first dimension corresponding to the RGB channels, and normalised to be used by ResNet.
        """
        
        # put the colour channel in front and normalise into range [0,1]
        if len(img.shape) == 3:
            norm_img = torch.moveaxis(img, -1, 0)/self.MAX_PIXEL_VALUE
        else:
            norm_img = torch.moveaxis(img, -1, 1)/self.MAX_PIXEL_VALUE
        
        # resize
        if dim is not None:
            assert dim >= self.RESNET_DIM, f"Provided dimension {dim} is less than the required for RESNET ({self.RESNET_DIM})"
            norm_img = T.Resize(dim)(norm_img)  
        else:
            norm_img = T.Resize(self.RESNET_DIM)(norm_img)
        
        # normalise mean and variance
        mean = torch.Tensor([0.485, 0.456, 0.406])
        std = torch.Tensor([0.229, 0.224, 0.225])
        
        return T.Normalize(mean = mean, std = std)(norm_img)
    
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
    
    def forward(self, x):
        """
        A forward pass through the network
        """
        
        if x.shape[:2] != (self.RESNET_DIM, self.RESNET_DIM):
            x = self.img_to_resnet(x)
            
        return self.model.encode(x)
        
    
    def compile_optimiser(self, **kwargs):
        """
        Sets the optimiser parameters.
        """
        self.optimiser = optim.Adam(self.parameters(), **kwargs)

    def update_checkpoint(self, checkpoint_dir, **checkpoint_data):
        """
        Updates the checkpoint dictionary.
        """
        for k, v in checkpoint_data.items():
            if k in self.checkpoint:
                self.checkpoint[k] = v

        if checkpoint_dir is not None:
            torch.save(self.checkpoint, checkpoint_dir)

    def update_checkpoint(self, **checkpoint_data):
        """
        Updates the checkpoint dictionary.
        """
        for k,v in checkpoint_data:
            if k in self.checkpoint:
                self.checkpoint[k] = v
    
    def train(self, dataloader, epochs, checkpoint_dir = None, transform = None, batch_log_rate = 100):
        """
        Trains the network.
        """

        self.to(self.device)

        for epoch in range(epochs):
            batch_losses = []
            for batch, (x_1,x_2) in enumerate(dataloader):
                # x_1 and x_2 are tensors containing patches, 
                # such that x_1[i] and x_2[i] are patches for the same area
                
                self.optimiser.zero_grad()

                x_1, x_2 = transform(x_1.to(self.device)), transform(x_2.to(self.device))
                
                z_1 = self.model(x_1)
                z_2 = self.model(x_2)
                
                z_batch = torch.stack((z_1,z_2), dim = 1).view(-1, self.OUTPUT_DIM)

                loss = self.contrastive_loss(z_batch)

                batch_losses.append(loss.cpu())
                
                loss.backward()
                self.optimiser.step()

                if batch % (len(dataloader) // batch_log_rate + 1) == 0:
                    with torch.no_grad():
                        avg_loss = torch.mean(batch_losses[-20:])
                        print(f"Epoch {epoch + 1}: [{batch + 1}/{len(dataloader)}] ---- NT-XENT = {avg_loss}")

                        self.update_checkpoint(checkpoint_dir = checkpoint_dir,
                                               epoch = epoch,
                                               batch = batch,
                                               model_state_dict = self.state_dict(),
                                               optimiser_state_dict = self.optimiser.state_dict,
                                               loss = loss.cpu(),
                                               avg_loss_20 = avg_loss,
                                               run_end = datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
                
        with torch.no_grad():
            self.update_checkpoint(epoch = epochs,
                                   batch = len(dataloader),
                                   model_state_dict = self.state_dict(),
                                   optimiser_state_dict = self.optimiser.state_dict,
                                   loss = loss.cpu(),
                                   avg_loss_20 = torch.mean(batch_losses[-20:]),
                                   run_end = datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

        return self.checkpoint

class WBMapSIMCLR(MapSIMCLR):
    """
    A SIMCLR model with Weights & Biases integration.
    """

    def __init__(self, encoder, encoder_layer_idx, projector_parameters, tau):

        super(WBMapSIMCLR, self).__init__(encoder, encoder_layer_idx, projector_parameters, tau)

    def train(self, dataloader, epochs, checkpoint_dir=None, transform=None, batch_log_rate=100):
        """
        Trains the network.
        ---------------------------------------------------------------------------------------------------
        :param dataloader: a PyTorch DataLoader, containing the training data.
        :param epochs: an int, the number of epochs for training.
        :param checkpoint_dir: a string, the directory to which to write the checkpoints.
        :param transform: a transformation function for the inputs, to apply right before passing it to the
                          network. By default no transformation is applied.
        :param batch_log_rate: an int. Every batch_log_rate batches, the performance of the network
                               is logged. By default logging is performed every 100 batches.

        """

        wandb.init(project="honours-project",
                   name=f"slurm_experiment_{time.strftime('%Y%m%d_%H%M%S')}",
                   config={
                       "epochs": epochs,
                       "batch_size": dataloader.batch_size,
                       "learning_rate": self.optimiser.param_groups[-1]['lr'],
                       "architecture": "SIMCLR (RESNET Encoder)"
                   })

        self.to(self.device)

        for epoch in range(epochs):
            batch_losses = []
            for batch, (x_1, x_2) in enumerate(dataloader):
                # x_1 and x_2 are tensors containing patches,
                # such that x_1[i] and x_2[i] are patches for the same area

                self.optimiser.zero_grad()

                x_1, x_2 = transform(x_1.to(self.device)), transform(x_2.to(self.device))

                z_1 = self.model(x_1)
                z_2 = self.model(x_2)

                z_batch = torch.stack((z_1, z_2), dim=1).view(-1, self.OUTPUT_DIM)
                loss = self.contrastive_loss(z_batch)

                batch_losses.append(loss.cpu())

                loss.backward()
                self.optimiser.step()

                if batch % (len(dataloader) // batch_log_rate + 1) == 0:
                    with torch.no_grad():
                        avg_loss = torch.mean(batch_losses[-20:])
                        print(f"Epoch {epoch + 1}: [{batch + 1}/{len(dataloader)}] ---- NT-XENT = {avg_loss}")

                        self.update_checkpoint(checkpoint_dir=checkpoint_dir,
                                               epoch=epoch,
                                               batch=batch,
                                               model_state_dict=self.state_dict(),
                                               optimiser_state_dict=self.optimiser.state_dict,
                                               loss=loss.cpu(),
                                               avg_loss_20=avg_loss,
                                               run_end=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

                        metrics = {"BYOL/train_loss": self.checkpoint["loss"],
                                   "BYOL/avg_20_train_loss": self.checkpoint["avg_loss_20"]}

                        wandb.log(metrics)

        with torch.no_grad():
            self.update_checkpoint(checkpoint_dir=checkpoint_dir,
                                   epoch=epochs,
                                   batch=len(dataloader),
                                   model_state_dict=self.state_dict(),
                                   optimiser_state_dict=self.optimiser.state_dict,
                                   loss=loss.cpu(),
                                   avg_loss_20=torch.mean(batch_losses[-20:]),
                                   run_end=datetime.now().strftime("%d/%m/%Y %H:%M:%S"))

            metrics = {"BYOL/train_loss": self.checkpoint["loss"],
                       "BYOL/avg_20_train_loss": self.checkpoint["avg_loss_20"]}

            wandb.log(metrics)

        wandb.finish()

        return self.checkpoint
