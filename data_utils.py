import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import os 
import librosa

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

def genSpoof_list_2021(dir_meta, track):
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    for line in l_meta:
        if track == "DF":
            _, key, _, _, _, _, _, _, _, _, _, _, _ = line.strip().split(" ")
        elif track == "LA":
            _, key, _, _, _, _, _, _ = line.strip().split(" ")
        #key = line.strip()
        file_list.append(key)
    return file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len > max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X, _ = librosa.load(str(self.base_dir / f"flac/{key}.flac"), sr=16000)
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X, _ = librosa.load(str(self.base_dir / f"flac/{key}.flac"), sr=16000)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key

class Dataset_ASVspoof2021_DF(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        # X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X, sr = librosa.load(str(self.base_dir / f"flac/{key}.flac"), sr=16000)
        # print(sr)
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key

def FoR_protocol_parser(path):
    file_list = []
    label_list = []

    protocol_f = open(path, 'r')
    lines = protocol_f.readlines()
    for line in lines:
        line = line.strip().split(' ')
        file_list.append(line[1])
        label_list.append(0 if line[4]=='spoof' else 1)

    return file_list, label_list

class Dataset_FoR(Dataset):
    def __init__(self, base_dir, protocol_path, is_train=False):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.base_dir = base_dir
        self.is_train = is_train
        self.protocol_path = protocol_path
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.files, self.labels = FoR_protocol_parser(self.protocol_path)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        key = self.files[index]
        X, _ = sf.read(os.path.join(self.base_dir, f"{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[index]
        if self.is_train:
            return x_inp, y
        else:
            return x_inp, key

def ITW_protocol_parser(path):
    file_list = []
    label_list = []

    protocol_f = open(path, 'r')
    lines = protocol_f.readlines()
    lines = lines[1:-1]
    for line in lines:
        line = line.strip().split(',')
        file_list.append(line[0])
        label_list.append(0 if line[2]=='spoof' else 1)

    return file_list, label_list

class Dataset_ITW(Dataset):
    def __init__(self, base_dir, is_train=False):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.base_dir = base_dir
        self.is_train = is_train
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.protocol_path = os.path.join(base_dir, 'meta.csv')
        self.files, self.labels = ITW_protocol_parser(self.protocol_path)
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        key = self.files[index]
        X, _ = sf.read(os.path.join(self.base_dir, 'data', f"{key}"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[index]
        if self.is_train:
            return x_inp, y
        else:
            return x_inp, key, y