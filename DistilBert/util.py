import torch
import numpy as np
import random
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def setup_seed(seed):
    print("current seed ", seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # if torch.cuda.is_available():
    #     if not cuda:
    #         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    #     else:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False








