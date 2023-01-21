import PIL
import os
import logging
import pickle as pk
from copy import deepcopy
from datetime import datetime
import time

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format='%(asctime)s %(levelname)-4s %(message)s',
                    level=logging.INFO,
                    datefmt='%d-%m-%Y %H:%M:%S')

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

PIL.Image.MAX_IMAGE_PIXELS = 933120000

from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, ResNet50_Weights

import wandb

from patch_dataset import CLPatchDataset

SCRATCH_DIR = "disk/scratch_big/s1908368"
DATASET_DIR = os.path.join(SCRATCH_DIR, "data/patch_dataset.pk")

cl_patch_dataset = None

# create the DataSet object (or load it if available)
if os.path.isfile(DATASET_DIR):
    with open(DATASET_DIR, "rb") as f:
        cl_patch_dataset = pk.load(f)
else:
    cl_patch_dataset = CLPatchDataset.from_dir(os.path.join(SCRATCH_DIR, "data/originals"), 64, verbose = True)
    cl_patch_dataset.save(DATASET_DIR)
    
# create the DataLoader object
cl_patch_loader = DataLoader(cl_patch_dataset, batch_size = 50, shuffle = True)

# create the BYOL model
projector_parameters = {"input_dim" : 2048, 
                        "hidden_dim" : 4096,
                        "output_dim" : 256,
                        "activation" : nn.ReLU(),
                        "use_bias" : True,
                        "use_batch_norm" : True}

predictor_parameters = {"input_dim" : 256, 
                        "hidden_dim" : 1024,
                        "output_dim" : 256,
                        "activation" : nn.ReLU(),
                        "use_bias" : True,
                        "use_batch_norm" : True}

encoder = resnet50(weights=ResNet50_Weights.DEFAULT)

wb_byol_nn = WBMapBYOL(encoder = encoder, 
                       encoder_layer_idx = -2,
                       projector_parameters = projector_parameters,
                       predictor_parameters = predictor_parameters,
                       ema_tau = 0.99)

wb_byol_nn.compile_optimiser()

logging.info(f"Using device: {wb_byol_nn.device}")

# train the model
wb_byol_nn.train(dataloader = cl_patch_loader, 
                 epochs = 2, 
                 checkpoint_dir = os.path.join(SCRATCH_DIR, "byol_checkpoint.pt"),
                 transform = wb_byol_nn.img_to_resnet, 
                 batch_log_rate = 50)
                                               