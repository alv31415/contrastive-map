import os
import logging
import pickle as pk
import argparse
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

logger = logging.getLogger()
logger.setLevel(logging.INFO)

logging.basicConfig(format="%(asctime)s %(levelname)-4s %(message)s",
                    level=logging.INFO,
                    datefmt="%d-%m-%Y %H:%M:%S")

logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

logging.info(f"Running main & importing modules...")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.resnet import resnet50, ResNet50_Weights

import wandb

from patch_dataset import CLPatchDataset
from byol import WBMapBYOL, MapBYOL
from simclr import MapSIMCLR

# --------------------------------------------------- PARSER ---------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument("-b", "--batch-size", type=int, default=50, metavar="N",
                        help="input batch size for training (default: 50)")
    parser.add_argument("-p", "--patch-size", type=int, default=64, metavar="N",
                        help="size of patches for dataset (default: 64)")
    parser.add_argument("-e", "--epochs", type=int, default=15, metavar="N",
                        help="number of epochs to train (default: 15)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("-s", "--seed", type=int, default=23, metavar="S",
                        help="random seed (default: 23)")
    parser.add_argument("-l", "--log-interval", type=int, default=50, metavar="N",
                        help="how many batches to wait before logging "
                             "training status (default: 50)")
    parser.add_argument("-i", "--input", required=True, help="Path to the "
                                                             "input data for the model to read")
    parser.add_argument("-o", "--output", required=True, help="Path to the "
                                                              "directory to write output to")
    parser.add_argument("-t", "--byol-ema-tau", type=float, default=0.99, metavar="TAU",
                        help="tau for BYOL's exponential moving average (default: 0.99)")
    parser.add_argument("-d", "--debug", action='store_true', default=False,
                        help="disables model running to debug inputs")

    return parser

# --------------------------------------------------- MAIN ---------------------------------------------------

def main(args):

    torch.manual_seed(args.seed)

    DATASET_DIR = os.path.join(args.input, "patch_dataset.pk")

    # create the DataSet object (or load it if available)
    if os.path.isfile(DATASET_DIR):
        with open(DATASET_DIR, "rb") as f:
            cl_patch_dataset = pk.load(f)
    else:
        cl_patch_dataset = CLPatchDataset.from_dir(map_directory = args.input,
                                                   patch_width = args.patch_size,
                                                   verbose=True)
        cl_patch_dataset.save(DATASET_DIR)

    # create the DataLoader object
    cl_patch_loader = DataLoader(cl_patch_dataset, batch_size = args.batch_size, shuffle = True)

    # create the BYOL model
    projector_parameters = {"input_dim": 2048,
                            "hidden_dim": 4096,
                            "output_dim": 256,
                            "activation": nn.ReLU(),
                            "use_bias": True,
                            "use_batch_norm": True}

    predictor_parameters = {"input_dim": 256,
                            "hidden_dim": 1024,
                            "output_dim": 256,
                            "activation": nn.ReLU(),
                            "use_bias": True,
                            "use_batch_norm": True}

    encoder = resnet50(weights=ResNet50_Weights.DEFAULT)

    wb_byol_nn = WBMapBYOL(encoder=encoder,
                           encoder_layer_idx=-2,
                           projector_parameters=projector_parameters,
                           predictor_parameters=predictor_parameters,
                           ema_tau = args.byol_ema_tau)

    wb_byol_nn.compile_optimiser()

    logging.info(f"Using device: {wb_byol_nn.device}")

    # train the model
    wb_byol_nn.train(dataloader = cl_patch_loader,
                     epochs = args.epochs,
                     checkpoint_dir = os.path.join(args.output, "byol_checkpoint.pt"),
                     transform = wb_byol_nn.img_to_resnet,
                     batch_log_rate = args.log_interval)

# --------------------------------------------------- RUN ---------------------------------------------------

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(f"Parsing arguments...")
    if not args.debug:
        main(args)
    else:
        print(args)

