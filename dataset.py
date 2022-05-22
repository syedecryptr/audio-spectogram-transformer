import os
import random

import torch
import torchaudio
from utils import *
from torch.utils.data import Dataset, DataLoader
from train_config import *


class TIDataset(Dataset):
    def __init__(self, mode="train"):
        self.file_names = os.listdir(WAV_FILES_PATH)
        random.shuffle(self.file_names)
        self.file_names = self.file_names[
                          :int(len(self.file_names) * DATASET_PERCENTAGE)]
        if mode == "train":
            self.file_names = self.file_names[
                              :int(len(self.file_names) * TRAIN_TEST_SPLIT)]
        else:
            self.file_names = self.file_names[
                              int(len(self.file_names) * TRAIN_TEST_SPLIT):]
        self.labels = [int(i.split("_")[0]) for i in self.file_names]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(
            os.path.join(WAV_FILES_PATH, self.file_names[idx]))
        window_len = int(sr * WINDOW_SIZE)  # 200
        stride_len = int(sr * STRIDE)  # 80
        desired_size = 64 * stride_len  # no of time frames = signal_size / stride (or hop_len)
        # desired_size = 64 * 80
        waveform = cut_if_necessary(waveform, desired_size)
        waveform = pad_if_necessary(waveform, desired_size)
        n_fft = window_len * 2 - 1
        spectogram = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                       hop_length=stride_len)(
            waveform)
        # no_of_time_frame(x_axis) = signal_size / stride = 64 * 80 / 80
        # no_of_freq(y_axis) = n_fft / 2 + 1 = window_len
        # spectogram shape : [1, window, signal_size / stride] # [1, y, x]
        # mel_spectogram_shape : [1, mel_filters, signal_size / stride] # [1, y, x]
        mel_spectogram = torchaudio.transforms.MelScale(n_mels=32,
                                                        n_stft=n_fft // 2 + 1)(
            spectogram)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spectogram, label


if __name__ == '__main__':
    training_data = TIDataset("eval")
    training_data_loader = DataLoader(training_data, batch_size=2, shuffle=True)
    img, label = next(iter(training_data_loader))
    print(len(training_data_loader), label.shape,
          img.shape)
