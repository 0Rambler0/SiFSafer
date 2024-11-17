# What's the Real: A Novel Design Philosophy for Robust AI-Synthesized Voice Detection

This repository contains our implementation of the paper published In Proceedings of the 32nd ACM International Conference on Multimedia (MM'24), "What's the Real: A Novel Design Philosophy for RobustAI-Synthesized Voice Detection".  In this paper, we propose a novel AI-synthesized voice detection framework named SiFSafer，which is robust to Speaker-irrelative Features (SiFs) and strongly resistant to existing attacks in the ASVspoof datasets  if the SiFs like silence segments are removed.

[Paper link here](https://doi.org/10.1145/3664647.3681100)

## Installation

First, clone the repository locally, create and activate a conda environment, and install the requirements :

```
$ git clone https://github.com/TakHemlata/SSL_Anti-spoofing.git
$ conda create -n sifsafer python==3.10.0
$ pip install "s3prl[all]"
$ apt install sox
```

## Experiments

### Dataset

Our experiments are performed on the logical access (LA) and deepfake (DF) partition of the ASVspoof  dataset (train on ASVspoof2019 LA dataset and evaluate on ASVspoof2019 LA, ASVspoof2021 LA, and ASVspoof2021 DF database).

The ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

The ASVspoof 2021 database is released on the zenodo site.

LA [here](https://zenodo.org/record/4837263#.YnDIinYzZhE)

DF [here](https://zenodo.org/record/4835108#.YnDIb3YzZhE)

For ASVspoof 2021 dataset keys (labels) and metadata are available [here](https://www.asvspoof.org/index2021.html)

## Evaluation

### Edit config file

To train the model run:

```

{   "database_path": your_dataset_path,
    "asv_score_path": asv_score_path
    "model_path": evaluation model weight path,
    "batch_size": 64,
    "num_epochs": 50,
    "lora_end_epoch": ,
    "loss": "CCE",
    "track": "LA",
    "type": the suffix of dataset,
    "eval_all_best": "True",
    "eval_output": "eval_scores.txt",
    "cudnn_deterministic_toggle": "True",
    "cudnn_benchmark_toggle": "False",
    "model_config": {
        "architecture": "SiFSafer",
        "inject_modules":["fc2", "out_proj"], # lora adapter injection modules
        "lora_layers":[0,12], # lora injection target layers
        "layers":[12,25], # feaures selection layers
        "tuning_layers":[0,12]
    },
    "optim_config": {
        "optimizer": "adam", 
        "lr": 0.000001,
        "weight_decay": 0.0001
    }
}
```

## Data Pre-Processing

SiFSafer is trained with the ASVSpoof2019 after removing silence segments. You can do the following to remove silence segments: 

```
1. Changing the path and dataset information of sox_silence.py

2. python3 sox_silence.py
```



### Tranining

To train your own model on ASVspoof2019 dataset:

```
python3 main_lora_end.py --config <your config path> --sr_model <upstream model path> --comment <save path suffix>

# our training parameter
python3 main_lora_end.py --config ./config/sifsafer.conf --sr_model xls_r_300m --comment <save path suffix>
```

### Evaluation

#### Evaluation on ASVspoof2019

Edit the "model_path" in config file and run:

```
python3 main_lora_end.py --config <your config path> --sr_model <upstream model path> --comment <save path suffix> --eval
```

#### Evaluation on ASVspoof2021

Edit the "model_path" in config file and run:

```
python3 eval_2021df.py  --config <your config path> --sr_model <upstream model path> --comment <save path suffix> --eval --eval_model_weights <evaluation weights> --track <LA or DF>
```

## Results using trained model:

EER: 4.85%, min t-DCF: 0.1228  on ASVspoof2019 LA  track.

EER: 12.69%, min t-DCF: 0.5023 on ASVspoof2021 LA track.

EER: 4.70% on ASVspoof2021 DF track.

Average EER: 7.41%

Please note that all of the result is evaluated with the data after removing the silence segments. 

## Contact

For any query regarding this repository, please contact:

* Xuan Hai: haix2024@lzu.edu.cn
* Xin Liu: bird@lzu.edu.cn

## Citation

If you use this code in your research please use the following citation:

```bibtex
@inproceedings{hai2024s,
  title={What's the Real: A Novel Design Philosophy for Robust AI-Synthesized Voice Detection},
  author={Hai, Xuan and Liu, Xin and Tan, Yuan and Liu, Gang and Li, Song and Niu, Weina and Zhou, Rui and Zhou, Xiaokang},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={6900--6909},
  year={2024}
}
```

## Our Other Works about Fake Voice Detection

Paper:
*  (ISSRE 2023) [Hidden-in-Wave: A Novel Idea to Camouflage AI-Synthesized Voices Based on Speaker-Irrelative Features](https://ieeexplore.ieee.org/abstract/document/10301243/)

* (ACM MM2023) [SiFDetectCracker: An Adversarial Attack Against Fake Voice Detection Based on Speaker-Irrelative Features](https://dl.acm.org/doi/abs/10.1145/3581783.3613841)

* (ICME 2024) [Ghost-in-Wave: How Speaker-Irrelative Features Interfere DeepFake Voice Detectors](https://ieeexplore.ieee.org/abstract/document/10688273/)


Open Source Code:

* SiFDetectCracker: https://github.com/0Rambler0/SiFDetectCracker.git

Industry Conference:

* (Black Hat USA 2022) [Human or Not - Can You Really Detect the Fake Voices?](https://www.researchgate.net/publication/362727059_Human_or_Not_Can_You_Really_Detect_the_Fake_Voices) 
* (HITB 2024)[Yes, I Am Human: Breaking Fake Voice Detection with Speaker-lrrelative Features]( https://conference.hitb.org/hitbsecconf2024bkk/commsec-track/)


