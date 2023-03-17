import argparse
from datetime import datetime


def get_parser():
    parser = argparse.ArgumentParser(description="Generate experiments for SLURM jobs")

    parser.add_argument("-i", "--main", required=True,
                        help="path to the main.py to be run by SLURM.")
    parser.add_argument("-d", "--scratch-data-dir", required=True,
                        help="path to the data in SLURM for training.")
    parser.add_argument("-o", "--scratch-out-dir", required=True,
                        help="path to the output folder in SLURM during training.")

    return parser


def get_experiment_name(use_byol, encoder, epochs, batch_size, tau, patch_size, pretrained, lr, use_geo_contrastive):
    string_tau = str(tau).replace(".", "_").replace(",", "_")
    string_tau += "0" if len(string_tau) == 3 else ""
    string_lr = str(lr).replace(".", "_").replace(",", "_")
    return f"{'g' if use_geo_contrastive else ''}" \
           f"{'b' if use_byol else 's'}" \
           f"-{'p' if pretrained else ''}{encoder}" \
           f"-e{epochs}" \
           f"-b{batch_size}" \
           f"-t{string_tau}" \
           f"-lr{string_lr}" \
           f"-p{patch_size}"


def get_default_arg_dict(scratch_data_dir, scratch_out_dir):
    return {
        "--batch-size": 32,
        "--patch-size": 64,
        "--epochs": 1,
        "--seed": 23,
        "--lr": 1e-3,
        "--logs-per-epoch": 500,
        "--evaluations-per-epoch": 100,
        "--train-proportion": 0.80,
        "--validation-proportion": 0.10,
        "--input": scratch_data_dir,
        "--output": scratch_out_dir,
        "--experiment-name": "b-r18-e1-b64-t0_99-p64",
        "--use-byol": True,
        "--use-geo-contrastive": False,
        "--encoder": "resnet18",
        "--pretrain-encoder": True,
        "--encoder-layer-idx": -2,
        "--byol-ema-tau": 0.99,
        "--simclr-tau": 0.99
    }


def create_experiment(main_file, scratch_data_dir, scratch_out_dir, experiment_args):
    arg_dict = get_default_arg_dict(scratch_data_dir, scratch_out_dir)

    for arg, value in experiment_args.items():
        if arg in arg_dict:
            if type(value) == bool:
                if not value:
                    arg_dict.pop(arg)
                else:
                    arg_dict[arg] = value
            else:
                arg_dict[arg] = value
        else:
            raise ValueError(f"The provided argument {arg} isn't a valid experiment argument.")

    arg_dict["--experiment-name"] = get_experiment_name(use_byol=arg_dict.get("--use-byol", False),
                                                        encoder=arg_dict["--encoder"],
                                                        epochs=arg_dict["--epochs"],
                                                        batch_size=arg_dict["--batch-size"],
                                                        tau=arg_dict["--byol-ema-tau" if arg_dict.get("--use-byol",
                                                                                                      False) else "--simclr-tau"],
                                                        patch_size=arg_dict["--patch-size"],
                                                        lr=arg_dict["--lr"],
                                                        pretrained=arg_dict.get("--pretrain-encoder", False),
                                                        use_geo_contrastive=arg_dict.get("--use-geo-contrastive", False))

    python_call = f"python {main_file}"

    for arg, value in arg_dict.items():
        if type(value) == str:
            python_call += f' {arg} "{value}"'
        elif type(value) == bool:
            python_call += f' {arg}'
        else:
            python_call += f' {arg} {value}'

    return python_call


def main(args):
    # current run

    # previous runs

    BATCH_SIZE = 64
    PATCH_SIZE = 128
    EPOCHS = 25
    SEED = 23
    LOGS_PER_EPOCH = 500
    EVALUATIONS_PER_EPOCH = 100
    TRAIN_PROPORTION = 0.8
    EVALUATION_PROPORTION = 0.1
    USE_GEO_CONTRASTIVE = False
    ENCODER_LAYER_IDX = -2
    LR = 1e-3

    experiment_argss = [
        # 1) simclr, pre-trained resnet18, e = 25, batch size = 64, temperature = 0.99
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 2) byol, pre-trained resnet18, e = 25, batch size = 64, ema tau= 0.99
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 3) simclr, pre-trained resnet18, e = 25, batch size = 64, temperature = 0.95
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.95
        },
        # 4) byol, pre-trained resnet18, e = 25, batch size = 64, ema tau= 0.95
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.95,
            "--simclr-tau": 0.99
        },
        # 5) simclr, pre-trained resnet18, e = 25, batch size = 64, temperature = 0.9
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.90
        },
        # 6) byol, pre-trained resnet18, e = 25, batch size = 64, ema tau= 0.9
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.90,
            "--simclr-tau": 0.99
        },
        # 7) simclr, pre-trained resnet18, e = 25, batch size = 64, temperature = 0.8
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.80
        },
        # 8) byol, pre-trained resnet18, e = 25, batch size = 64, ema tau= 0.8
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.80,
            "--simclr-tau": 0.99
        },
        # 9) simclr, not pre-trained resnet18, e = 25, batch size = 64, temperature= 0.99
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": False,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 10) byol, not pre-trained resnet18, e = 25, batch size = 64, ema tau= 0.99
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": False,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 11) simclr, pre-trained resnet34, e = 25, batch size = 64, temperature= 0.99
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet34",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 12) byol, pre-trained resnet34, e = 25, batch size = 64, ema tau= 0.99
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet34",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 13) simclr, cnn, e = 25, batch size = 64, temperature= 0.99
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "cnn",
            "--pretrain-encoder": False,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 14) byol, cnn, e = 25, batch size = 64, ema tau = 0.99
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": PATCH_SIZE,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "cnn",
            "--pretrain-encoder": False,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 15) simclr, resnet18, e = 25, batch size = 64, temperature = 0.99, patch size = 224
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": 224,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 16) byol, resnet18, e = 25, batch size = 64, ema tau = 0.99, patch size = 224
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": 224,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": LR,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": False,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 17) simclr, resnet18, e = 25, batch size = 64, temperature = 0.99, lr = 1e-2
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": 1e-2,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 18) byol, resnet18, e = 25, batch size = 64, ema tau = 0.99, lr = 1e-2
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": 1e-2,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": USE_GEO_CONTRASTIVE,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        """
        # 19) simclr, resnet18, e = 25, batch size = 64, temperature = 0.99, geo-contrastive objective
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": 1e-3,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": False,
            "--use-geo-contrastive": True,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        },
        # 20) byol, resnet18, e = 25, batch size = 64, ema tau = 0.99, geo-contrastive objective
        {
            "--batch-size": BATCH_SIZE,
            "--patch-size": 128,
            "--epochs": EPOCHS,
            "--seed": SEED,
            "--lr": 1e-3,
            "--logs-per-epoch": LOGS_PER_EPOCH,
            "--evaluations-per-epoch": EVALUATIONS_PER_EPOCH,
            "--train-proportion": TRAIN_PROPORTION,
            "--validation-proportion": EVALUATION_PROPORTION,
            "--use-byol": True,
            "--use-geo-contrastive": True,
            "--encoder": "resnet18",
            "--pretrain-encoder": True,
            "--encoder-layer-idx": ENCODER_LAYER_IDX,
            "--byol-ema-tau": 0.99,
            "--simclr-tau": 0.99
        }
        """
    ]

    """
    SLRUM JOB ID: 1507975
    experiment_argss = [{"--epochs": 5, "--patch-size": 64, "--batch-size" : 32},
                    {"--epochs": 5, "--patch-size": 32, "--batch-size" : 32},
                    {"--epochs": 5, "--patch-size": 128, "--batch-size" : 32}
                    ]

    SLRUM JOB ID: 1507981
    experiment_argss = [{"--epochs": 5, "--use-byol": False, "--encoder-layer-idx": -2},
                        {"--epochs": 5, "--pretrain-encoder": False},
                        {"--epochs": 5, "--pretrain-encoder": False, "--encoder": "resnet34"},
                        {"--epochs": 5, "--pretrain-encoder": False, "--encoder": "cnn", "--encoder-layer-idx": -1},
                        {"--epochs": 5, "--encoder": "resnet34"},
                        {"--epochs": 5, "--byol-ema-tau": 0.95},
                        {"--epochs": 5, "--byol-ema-tau": 0.9},
                        {"--epochs": 5, "--byol-ema-tau": 0.8}
                        ]
    """
    """
    SLURM JOB ID: 1512856
    experiment_argss = [{"--epochs": 5, "--use-byol": False, "--encoder-layer-idx": -2, "--patch-size": 128},
                        {"--epochs": 5, "--encoder": "cnn", "--encoder-layer-idx": -1, "--patch-size": 128},
                        {"--epochs": 5, "--encoder": "resnet34", "--encoder-layer-idx": -2, "--patch-size": 128},
                        {"--epochs": 5, "--encoder": "resnet50", "--encoder-layer-idx": -2, "--patch-size": 128}
                        ]
    """
    """
    1519859
    experiment_argss = [{"--epochs": 5, "--use-byol": False, "--encoder-layer-idx": -2, "--patch-size": 128, "--pretrain-encoder": False},
                        {"--epochs": 5, "--use-byol": True, "--encoder-layer-idx": -2, "--patch-size": 128, "--pretrain-encoder": False},
                        {"--epochs": 5, "--byol-ema-tau": 0.95, "--use-byol": True, "--encoder-layer-idx": -2, "--patch-size": 128, "--pretrain-encoder": True},
                        {"--epochs": 5, "--byol-ema-tau": 0.9, "--use-byol": True, "--encoder-layer-idx": -2, "--patch-size": 128, "--pretrain-encoder": True},
                        {"--epochs": 5, "--byol-ema-tau": 0.8, "--use-byol": True, "--encoder-layer-idx": -2, "--patch-size": 128, "--pretrain-encoder": True},
                        {"--epochs": 5, "--encoder": "cnn", "--encoder-layer-idx": -1, "--patch-size": 128},
                        ]
    """
    """
    experiment_argss = [{"--epochs": 5, "--encoder": "cnn", "--encoder-layer-idx": -1, "--patch-size": 128}]
    """

    experiment_run_args = [create_experiment(main_file=args.main,
                                             scratch_data_dir=args.scratch_data_dir,
                                             scratch_out_dir=args.scratch_out_dir,
                                             experiment_args=experiment_args) + "\n"
                           for experiment_args in experiment_argss]

    with open("experiments.txt", "w+") as f:
        f.writelines(experiment_run_args)

    with open("ran_experiments.txt", "a+") as f:
        f.write("\n")
        f.write("=" * 20 + datetime.now().strftime("%d/%m/%Y %H:%M:%S") + "=" * 20 + "\n")
        f.writelines(experiment_run_args)
        f.write("=" * 60)


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
