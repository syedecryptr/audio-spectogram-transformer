from train_config import *
import torch.nn.functional as F


def pad_if_necessary(waveform):
    if waveform.shape[1] < DESIRED_SIZE:
        missing_samples = DESIRED_SIZE - waveform.shape[1]
        waveform = F.pad(waveform, (0, missing_samples))
    return waveform


def cut_if_necessary(waveform):
    if waveform.shape[1] > DESIRED_SIZE:
        waveform = waveform[:, :DESIRED_SIZE]
    return waveform
