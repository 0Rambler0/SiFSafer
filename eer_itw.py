import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

# 读取文件数据
def read_data(file_path):
    scores = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            label = int(parts[1])
            score = float(parts[2])
            labels.append(label)
            scores.append(score)
    return np.array(labels), np.array(scores)

# 计算EER
def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    eer_threshold = interp1d(fpr, thresholds)(eer)
    return eer, eer_threshold

# 文件路径
file_path = 'your score file path'

# 读取数据
labels, scores = read_data(file_path)

# 计算EER
eer, eer_threshold = compute_eer(labels, scores)

print(f'EER: {eer}')
print(f'EER Threshold: {eer_threshold}')