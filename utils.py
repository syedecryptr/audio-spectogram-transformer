from train_config import *
import torch.nn.functional as F


def pad_if_necessary(waveform, desired_size):
    if waveform.shape[1] < desired_size:
        missing_samples = desired_size - waveform.shape[1]
        waveform = F.pad(waveform, (0, missing_samples))
    return waveform


def cut_if_necessary(waveform, desired_size):
    if waveform.shape[1] > desired_size:
        waveform = waveform[:, :desired_size]
    return waveform
