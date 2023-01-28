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
from torchvision.models.resnet import resnet18, ResNet18_Weights, resnet34, ResNet34_Weights, resnet50, ResNet50_Weights

from patch_dataset import CLPatchDataset
from byol import WBMapBYOL, MapBYOL
from simclr import MapSIMCLR
from cnn import CNN

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
    parser.add_argument("--encoder-layer-idx", type=int, default=-1, metavar="i",
                        help="index from which to take the output of the encoder (default: -1)")
    parser.add_argument("--simclr-tau", type=float, default=1.0, metavar="TAU",
                        help="tau for SimCLRs NTXENT loss (default: 1)")
    parser.add_argument("-d", "--debug", action='store_true', default=False,
                        help="disables model running to debug inputs")
    parser.add_argument("-m", "--use-byol", action='store_true', default=False,
                        help="if present, uses BYOL model, otherwise SimCLR")
    parser.add_argument("-x", "--experiment-name", help = "name of experiment, to store logs and outputs")
    parser.add_argument("-r", "--encoder", choices = ["cnn", "resnet18", "resnet34", "resnet50"],
                        help="type of encoder to use (out of cnn, resnet18, resnet50)")
    parser.add_argument("--pretrain-encoder", action = "store_true", default=False,
                        help = "whether to use pretrained weights in the encoder (only works for ResNet18 or ResNet50)")

    return parser

# --------------------------------------------------- MAIN ---------------------------------------------------

def main(args):

    torch.manual_seed(args.seed)

    DATASET_DIR = os.path.join(args.input, "patch_dataset.pk")
    logging.info(f"File at {DATASET_DIR}: {os.path.isfile(DATASET_DIR)}")

    # create the DataSet object (or load it if available)
    if os.path.isfile(DATASET_DIR):
        with open(DATASET_DIR, "rb") as f:
            cl_patch_dataset = pk.load(f)
    else:
        cl_patch_dataset = CLPatchDataset.from_dir(map_directory = args.input,
                                                   patch_width = args.patch_size,
                                                   verbose=True)
        cl_patch_dataset.save(DATASET_DIR)

    logging.info(f"File at {DATASET_DIR}: {os.path.isfile(DATASET_DIR)}")

    # create the DataLoader object
    cl_patch_loader = DataLoader(cl_patch_dataset, batch_size = args.batch_size, shuffle = True, num_workers = 4)

    use_transform = True
    match args.encoder:
        case "cnn":
            use_transform = False
            projector_parameters = {}
            encoder = CNN()
        case "resnet34":
            projector_parameters = {"input_dim": 512,
                                    "hidden_dim": 2048,
                                    "output_dim": 256,
                                    "activation": nn.ReLU(),
                                    "use_bias": True,
                                    "use_batch_norm": True}

            if args.pretrain_encoder:
                encoder = resnet34(weights = ResNet34_Weights.DEFAULT)
            else:
                encoder = resnet34()
        case "resnet50":
            projector_parameters = {"input_dim": 2048,
                                       "hidden_dim": 4096,
                                       "output_dim": 256,
                                       "activation": nn.ReLU(),
                                       "use_bias": True,
                                       "use_batch_norm": True}

            if args.pretrain_encoder:
                encoder = resnet50(weights = ResNet50_Weights.DEFAULT)
            else:
                encoder = resnet50()
        case _:
            projector_parameters = {"input_dim": 512,
                                       "hidden_dim": 2048,
                                       "output_dim": 256,
                                       "activation": nn.ReLU(),
                                       "use_bias": True,
                                       "use_batch_norm": True}

            if args.pretrain_encoder:
                encoder = resnet18(weights = ResNet18_Weights.DEFAULT)
            else:
                encoder = resnet18()

    logging.info(f"Using encoder {args.encoder} with pretrained weights = {args.pretrain_encoder}")

    # create the BYOL model
    predictor_parameters = {"input_dim": 256,
                            "hidden_dim": 1024,
                            "output_dim": 256,
                            "activation": nn.ReLU(),
                            "use_bias": True,
                            "use_batch_norm": True}

    if args.use_byol:
        model = MapBYOL(encoder=encoder,
                          encoder_layer_idx=args.encoder_layer_idx,
                          projector_parameters=projector_parameters,
                          predictor_parameters=predictor_parameters,
                          ema_tau = args.byol_ema_tau)

        logging.info(f"Using BYOL with tau = {args.byol_ema_tau}, with encoder layer index = {args.encoder_layer_idx}")
    else:
        model = MapSIMCLR(encoder=encoder,
                          encoder_layer_idx=args.encoder_layer_idx,
                          projector_parameters=projector_parameters,
                          tau = args.simclr_tau)

        logging.info(f"Using SimCLR with tau = {args.simclr_tau}, with encoder layer index = {args.encoder_layer_idx}")

    model.compile_optimiser()

    logging.info(f"Using device: {model.device}")

    #torch.cuda.empty_cache()

    # train the model
    model.train(dataloader = cl_patch_loader,
                epochs = args.epochs,
                checkpoint_dir = args.output,
                transform = model.img_to_resnet if use_transform else None,
                batch_log_rate = args.log_interval)

# --------------------------------------------------- RUN ---------------------------------------------------

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")
    if not args.debug:
        main(args)
    else:
        print(args)

