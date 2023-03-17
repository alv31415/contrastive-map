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
        "--simclr-tau": 0.99,
        "--patience-prop": 0.25
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
    BATCH_SIZE = 64
    PATCH_SIZE = 128
    EPOCHS = 25
    SEED = 23
    LOGS_PER_EPOCH = 130
    EVALUATIONS_PER_EPOCH = 100
    TRAIN_PROPORTION = 0.8
    EVALUATION_PROPORTION = 0.1
    USE_GEO_CONTRASTIVE = False
    ENCODER_LAYER_IDX = -2
    LR = 1e-3
    PATIENCE_PROP = 0.25

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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.95,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.90,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.80,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
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
            "--simclr-tau": 0.99,
            "--patience-prop": PATIENCE_PROP
        }]
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
        "--simclr-tau": 0.99,
        "--patience-prop": PATIENCE_PROP
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
        "--simclr-tau": 0.99,
        "--patience-prop": PATIENCE_PROP
    }
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
