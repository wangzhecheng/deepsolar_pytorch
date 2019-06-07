from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms, utils
import torchvision.transforms.functional as TF

from tqdm import tqdm
import numpy as np
import json
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import skimage
import skimage.io
import skimage.transform
from PIL import Image
import time
import os
from os.path import join, exists
import copy
import random
from collections import OrderedDict
from sklearn.metrics import r2_score

from torch.nn import functional as F
from torchvision.models import Inception3

from inception_modified import InceptionSegmentation
from image_dataset import ImageFolderModified

# Configuration
# directory for loading training/validation/test data
data_dir = '/home/ubuntu/projects/deepsolar/deepsolar_dataset_toy/test'
old_ckpt_path = '/home/ubuntu/projects/deepsolar/deepsolar_pytorch_pretrained/deepsolar_seg_pretrained.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_size = 299
batch_size = 1   # must be 1 for testing segmentation
threshold = 0.5  # threshold probability to identify am image as positive
level = 2

def metrics(stats):
    """stats: {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}
    return: must be a single number """
    precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    return 0.5*(precision + recall)


def test_model(model, dataloader, metrics, threshold):
    stats = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}
    model.eval()
    CAM_list = []
    for inputs, labels, paths in tqdm(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            _, outputs, CAM = model(inputs, testing=True)   # CAM is a 1 x 35 x 35 activation map
            prob = F.softmax(outputs, dim=1)
            preds = prob[:, 1] >= threshold

        CAM = CAM.squeeze(0).cpu().numpy()   # transform tensor into numpy array
        for i in range(preds.size(0)):
            predicted_label = preds[i]
            if predicted_label.cpu().item():
                CAM_list.append((CAM, paths[i]))        # only use the generated CAM if it is predicted to be 1
            else:
                CAM_list.append((np.zeros_like(CAM), paths[i]))  # otherwise the CAM is a totally black one

        stats['TP'] += torch.sum((preds == 1) * (labels == 1)).cpu().item()
        stats['TN'] += torch.sum((preds == 0) * (labels == 0)).cpu().item()
        stats['FP'] += torch.sum((preds == 1) * (labels == 0)).cpu().item()
        stats['FN'] += torch.sum((preds == 0) * (labels == 1)).cpu().item()

    metric_value = metrics(stats)
    return stats, metric_value, CAM_list

transform_test = transforms.Compose([
                 transforms.Resize(input_size),
                 transforms.ToTensor(),
                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                 ])

if __name__ == '__main__':
    # data
    dataset_test = ImageFolderModified(data_dir, transform_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=4)
    # model
    model = InceptionSegmentation(num_outputs=2, level=level)
    model.load_existing_params(old_ckpt_path)

    model = model.to(device)

    stats, metric_value, CAM_list = test_model(model, dataloader_test, metrics, threshold=threshold)
    precision = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FP'] + 0.00001)
    recall = (stats['TP'] + 0.00001) * 1.0 / (stats['TP'] + stats['FN'] + 0.00001)
    print('metric value: '+str(metric_value))
    print('precision: ' + str(round(precision, 4)))
    print('recall: ' + str(round(recall, 4)))

    with open('CAM_list.pickle', 'w') as f:
        pickle.dump(CAM_list, f)

