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
from geo_patch_dataset import GeoPatchDataset
from byol import MapBYOL
from simclr import MapSIMCLR
from geo_simclr import GeoMapSIMCLR
from cnn import CNN


# --------------------------------------------------- PARSER ---------------------------------------------------

def get_parser():
    parser = argparse.ArgumentParser(description="Self-Supervised Map Embeddings")

    # training params
    parser.add_argument("--batch-size", type=int, default=50, metavar="N",
                        help="input batch size for training (default: 50)")
    parser.add_argument("--patch-size", type=int, default=64, metavar="N",
                        help="size of patches for dataset (default: 64)")
    parser.add_argument("--epochs", type=int, default=15, metavar="N",
                        help="number of epochs to train (default: 15)")
    parser.add_argument("--lr", type=float, default=1e-3, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--seed", type=int, default=23, metavar="S",
                        help="random seed (default: 23)")
    parser.add_argument("--logs-per-epoch", type=int, default=50, metavar="N",
                        help="how many times to log training progress per epoch (default: 50)")
    parser.add_argument("--evaluations-per-epoch", type=int, default=50, metavar="N",
                        help="how many times to evaluate model per epoch (default: 50)")
    parser.add_argument("--patience-prop", type=float, default=0.25, metavar="N",
                        help="proportion of evaluations in which validation loss doesn't change, before applying stopping (default: 25)")
    parser.add_argument("--train-proportion", type=float, default=0.8, metavar="P",
                        help="proportion of data to be used for training (default: 0.8)")
    parser.add_argument("--validation-proportion", type=float, default=0.1,
                        help="proportion of data to be used for validation (default: 0.1)")

    # I/O params
    parser.add_argument("--input", required=True, help="Path to the "
                                                       "input data for the model to read")
    parser.add_argument("--output", required=True, help="Path to the "
                                                        "directory to write output to")
    parser.add_argument("--experiment-name", required=True, help="name of experiment, to store logs and outputs")

    # model choices
    parser.add_argument("--use-byol", action='store_true', default=False,
                        help="if present, uses BYOL model, otherwise SimCLR")
    parser.add_argument("--use-geo-contrastive", action='store_true', default=False,
                        help="if present, also uses a geographical contrastive objective")
    parser.add_argument("--encoder", choices=["cnn", "resnet18", "resnet34", "resnet50"],
                        help="type of encoder to use (out of cnn, resnet18, resnet50)")
    parser.add_argument("--pretrain-encoder", action="store_true", default=False,
                        help="whether to use pretrained weights in the encoder (only works for ResNet18 or ResNet50)")
    parser.add_argument("--encoder-layer-idx", type=int, default=-1, metavar="i",
                        help="index from which to take the output of the encoder (default: -1)")

    # model hyperparameters
    parser.add_argument("--byol-ema-tau", type=float, default=0.99, metavar="TAU",
                        help="tau for BYOL's exponential moving average (default: 0.99)")
    parser.add_argument("--simclr-tau", type=float, default=1.0, metavar="TAU",
                        help="tau for SimCLRs NTXENT loss (default: 1)")

    # debugging
    parser.add_argument("--debug", action='store_true', default=False,
                        help="disables model running to debug inputs")

    return parser


# --------------------------------------------------- MAIN ---------------------------------------------------

def main(args):
    torch.manual_seed(args.seed)

    logging.info(f"Using geographical contrastive objective: {args.use_geo_contrastive}")

    patch_train_dataset_name = f"patch_train_dataset_{args.patch_size}.pk"
    patch_validation_dataset_name = f"patch_validation_dataset_{args.patch_size}.pk"
    patch_test_dataset_name = f"patch_test_dataset_{args.patch_size}.pk"

    train_dataset_name = f"geo_{patch_train_dataset_name}" if args.use_geo_contrastive else patch_train_dataset_name
    validation_dataset_name = f"geo_{patch_validation_dataset_name}" if args.use_geo_contrastive else patch_validation_dataset_name
    test_dataset_name = f"geo_{patch_test_dataset_name}" if args.use_geo_contrastive else patch_test_dataset_name

    TRAIN_DATASET_DIR = os.path.join(args.input, train_dataset_name)
    VALIDATION_DATASET_DIR = os.path.join(args.input, validation_dataset_name)
    TEST_DATASET_DIR = os.path.join(args.input, test_dataset_name)

    logging.info(f"File at {TRAIN_DATASET_DIR}: {os.path.isfile(TRAIN_DATASET_DIR)}")
    logging.info(f"File at {VALIDATION_DATASET_DIR}: {os.path.isfile(VALIDATION_DATASET_DIR)}")
    logging.info(f"File at {TEST_DATASET_DIR}: {os.path.isfile(TEST_DATASET_DIR)}")

    # create the DataSet object (or load it if available)
    if os.path.isfile(TRAIN_DATASET_DIR) and os.path.isfile(VALIDATION_DATASET_DIR) and os.path.isfile(
            TEST_DATASET_DIR):
        with open(TRAIN_DATASET_DIR, "rb") as f:
            train_dataset = pk.load(f)

        with open(VALIDATION_DATASET_DIR, "rb") as f:
            validation_dataset = pk.load(f)

        with open(TEST_DATASET_DIR, "rb") as f:
            test_dataset = pk.load(f)
    else:

        if args.use_geo_contrastive \
                and os.path.isfile(os.path.join(args.input, patch_train_dataset_name)) \
                and os.path.isfile(os.path.join(args.input, patch_validation_dataset_name)) \
                and os.path.isfile(os.path.join(args.input, patch_test_dataset_name)):

            with open(os.path.join(args.input, patch_train_dataset_name), "rb") as f:
                train_dataset = pk.load(f)

            with open(os.path.join(args.input, patch_validation_dataset_name), "rb") as f:
                validation_dataset = pk.load(f)

            with open(os.path.join(args.input, patch_test_dataset_name), "rb") as f:
                test_dataset = pk.load(f)
        else:
            patch_dataset = CLPatchDataset.from_dir(map_directory=args.input,
                                                    patch_width=args.patch_size,
                                                    verbose=True)

            train_dataset, validation_dataset, test_dataset = patch_dataset.unique_split(p_train=args.train_proportion,
                                                                                         p_validation=args.validation_proportion,
                                                                                         seed=args.seed)

        if args.use_geo_contrastive:
            geographical_patch_dict, max_index = GeoPatchDataset.get_geographical_patch_dict([train_dataset,
                                                                                              validation_dataset,
                                                                                              test_dataset])

            train_dataset = GeoPatchDataset.from_dataset(patch_dataset=train_dataset,
                                                         shift=1,
                                                         n_contexts=1,
                                                         geographical_patch_dict=geographical_patch_dict,
                                                         max_index=max_index)
            validation_dataset = GeoPatchDataset.from_dataset(patch_dataset=validation_dataset,
                                                              shift=1,
                                                              n_contexts=1,
                                                              geographical_patch_dict=geographical_patch_dict,
                                                              max_index=max_index)
            test_dataset = GeoPatchDataset.from_dataset(patch_dataset=test_dataset,
                                                        shift=1,
                                                        n_contexts=1,
                                                        geographical_patch_dict=geographical_patch_dict,
                                                        max_index=max_index)

        train_dataset.save(TRAIN_DATASET_DIR)
        validation_dataset.save(VALIDATION_DATASET_DIR)
        test_dataset.save(TEST_DATASET_DIR)

    logging.info(f"Generated training dataset with {len(train_dataset)} samples.")
    logging.info(f"Generated validation dataset with {len(validation_dataset)} samples.")
    logging.info(f"Generated test dataset with {len(test_dataset)} samples.")

    # create the DataLoader object

    num_workers = 2

    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=num_workers)
    validation_loader = DataLoader(validation_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=num_workers)

    encoder_parameters = {"encoder_layer_idx": args.encoder_layer_idx,
                          "use_resnet": args.encoder != "cnn"}

    if args.encoder == "cnn":
        projector_parameters = {"input_dim": 512,
                                "hidden_dim": 2048,
                                "output_dim": 256,
                                "activation": nn.ReLU(),
                                "use_bias": True,
                                "use_batch_norm": True}
        encoder = CNN(input_dim=args.patch_size,
                      in_channels=3,
                      output_dim=512,
                      use_bias=True,
                      use_batch_norm=True,
                      use_flatten=False)
    elif args.encoder == "resnet34":
        projector_parameters = {"input_dim": 512,
                                "hidden_dim": 2048,
                                "output_dim": 256,
                                "activation": nn.ReLU(),
                                "use_bias": True,
                                "use_batch_norm": True}

        if args.pretrain_encoder:
            encoder = resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            encoder = resnet34()
    elif args.encoder == "resnet50":
        projector_parameters = {"input_dim": 2048,
                                "hidden_dim": 4096,
                                "output_dim": 256,
                                "activation": nn.ReLU(),
                                "use_bias": True,
                                "use_batch_norm": True}

        if args.pretrain_encoder:
            encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            encoder = resnet50()
    else:
        projector_parameters = {"input_dim": 512,
                                "hidden_dim": 2048,
                                "output_dim": 256,
                                "activation": nn.ReLU(),
                                "use_bias": True,
                                "use_batch_norm": True}

        if args.pretrain_encoder:
            encoder = resnet18(weights=ResNet18_Weights.DEFAULT)
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
                        encoder_parameters=encoder_parameters,
                        projector_parameters=projector_parameters,
                        predictor_parameters=predictor_parameters,
                        ema_tau=args.byol_ema_tau, )

        logging.info(f"Using BYOL with tau = {args.byol_ema_tau}, with encoder layer index = {args.encoder_layer_idx}")
    else:

        if args.use_geo_contrastive:
            model = GeoMapSIMCLR(encoder=encoder,
                                 encoder_parameters=encoder_parameters,
                                 projector_parameters=projector_parameters,
                                 tau=args.simclr_tau,
                                 sim_weight=0.7)

            logging.info(
                f"Using GeoSimCLR with tau = {args.simclr_tau}, with encoder layer index = {args.encoder_layer_idx}")
        else:
            model = MapSIMCLR(encoder=encoder,
                              encoder_parameters=encoder_parameters,
                              projector_parameters=projector_parameters,
                              tau=args.simclr_tau)

            logging.info(f"Using SimCLR with tau = {args.simclr_tau}, with encoder layer index = {args.encoder_layer_idx}")

    model.compile_optimiser(lr=args.lr)

    logging.info(f"Using device: {model.device}")

    torch.cuda.empty_cache()

    # train the model
    model.train_model(train_loader=train_loader,
                      validation_loader=validation_loader,
                      epochs=args.epochs,
                      checkpoint_dir=os.path.join(args.output, args.experiment_name),
                      logs_per_epoch=args.logs_per_epoch,
                      evaluations_per_epoch=args.evaluations_per_epoch,
                      patience_prop=args.patience_prop
                      )

# --------------------------------------------------- RUN ---------------------------------------------------

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logging.info(f"Parsed arguments: {args}")
    if not args.debug:
        main(args)
    else:
        print(args)
