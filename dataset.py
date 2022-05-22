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
        waveform = cut_if_necessary(waveform)
        waveform = pad_if_necessary(waveform)
        window = int(sr * WINDOW_SIZE // 1000)  # ms
        stride = int(sr * STRIDE // 1000)
        mel_spectogram = torchaudio.transforms.MelSpectrogram(n_fft=window,
                                                              hop_length=stride,
                                                              n_mels=FILTER_BANKS)(
            waveform)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return mel_spectogram, label


if __name__ == '__main__':
    training_data = TIDataset("eval")
    training_data_loader = DataLoader(training_data, batch_size=2, shuffle=True)
    label, img = next(iter(training_data_loader))
    print(len(training_data_loader), label.shape,
          img.shape)
