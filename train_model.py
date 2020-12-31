import argparse
import json
from datetime import datetime

import pytz
import torch

from supported_optimizers import optimizers
from unet import createAndTrainModel
from unet_dataset import LocalFilesUnetDataset

# ------------How to save a torch model depending on the actual goal
#            https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch

crits = {
    "L1Loss": torch.nn.L1Loss,
    "MSELoss": torch.nn.MSELoss,
    "BCELoss": torch.nn.BCELoss
}

parser = argparse.ArgumentParser()

# Dataset related
parser.add_argument("files_list",
                    help="Creates and train a noise reducing UNet Convolutionnal Neural Network, provided a list of reference images in any text file")

# Training related
parser.add_argument("--batch_size", help="Number of examples per batch", type=int, default=4)
parser.add_argument("--epochs", help="Number of training epochs", type=int, default=50)
parser.add_argument("--log_file_name", help="Path to a new file to log the results of training")
parser.add_argument("--save_frequency", help="How many epochs before saving a checkpoint", type=int, default=10)
parser.add_argument("--run_name", help="Just to give a name to the run, so you can quickly identify it")

# Loss function related
parser.add_argument("--criterion", help="Loss function to minimize for training", choices=crits.keys(),
                    default="L1Loss")

# Optimizer related
parser.add_argument("--optimizer_params", help="Path to a file with optimizer's parameters")
parser.add_argument("--learning_rate", help="Optimizer's learning rate", type=float, default=1e-3)

args = parser.parse_args()

if __name__ == "__main__":
    dataset = LocalFilesUnetDataset(args.files_list)
    createAndTrainModelArgs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "num_epochs": args.epochs,
        "criterion": crits[args.criterion],
        "save_frequency": args.save_frequency
    }
    if args.log_file_name:
        createAndTrainModelArgs["log_file_name"] = args.log_file_name

    if args.optimizer_params:
        optInfo = json.load(args.optimizer_params)
        optimizerFactory = optimizers[optInfo["name"]](optInfo["params"])
        createAndTrainModelArgs["optimizer"] = optimizerFactory

    if args.run_name:
        createAndTrainModelArgs["run_name"] = args.run_name

    model = createAndTrainModel(**createAndTrainModelArgs)
    timeString = datetime.now(pytz.timezone("CET")).strftime("%b-%d-%Hh%M")
    model.cpu()  # Before ever saving model to file, make sure it has host memory mapping (won't depend on harware)
