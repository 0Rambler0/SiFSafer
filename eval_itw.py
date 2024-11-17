
import argparse
import json
import os
import sys
import warnings
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_utils import (Dataset_ITW)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool

warnings.filterwarnings("ignore", category=FutureWarning)


def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    type = config["type"]

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    database_path = "your dataset path"

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}_{}_{}".format(
        "LA",
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"], type, args.sr_model)
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / "{}.txt".format(args.fname)
    os.makedirs(model_save_path, exist_ok=True)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    sr_model = args.sr_model
    model = get_model(model_config, sr_model, device)

    # define dataloaders
    eval_loader = get_loader(database_path, config)

    # evaluates pretrained model and exit script
    model.load_state_dict(torch.load(args.eval_model_weights, map_location=device))
    print("Model loaded : {}".format(args.eval_model_weights))
    print("Start evaluation...")
    produce_evaluation_file(eval_loader, model, device,
                            eval_score_path)
    print("DONE.")

def get_model(model_config: Dict, sr_model: str, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config, sr_model, device).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""

    # file_eval = genSpoof_list_2021(dir_meta=eval_trial_path, track=track)
    eval_set = Dataset_ITW(base_dir=database_path)
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=False,
                             drop_last=False,
                             pin_memory=True)

    return eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    device: torch.device,
    save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    fname_list = []
    score_list = []
    label_list = []
    for batch_x, utt_id, label in data_loader:
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_out = model(batch_x, torch.LongTensor([16000*4]*batch_size))
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())
        label_list.extend(label)
    with open(save_path, "w") as fh:
        for fn, sco, label in zip(fname_list, score_list, label_list):
            fh.write("{} {} {}\n".format(fn, label, sco))
    print("Scores saved to {}".format(save_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument("--sr_model",
                        type=str,
                        help="upstream model",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    parser.add_argument("--fname",
                        type=str,
                        default="eval_itw")
    main(parser.parse_args())
