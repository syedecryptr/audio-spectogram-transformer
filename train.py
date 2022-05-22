import random

import torch
import os
from dataset import TIDataset
from torch.utils.data import DataLoader
from models import VIT
from tqdm import tqdm
from train_config import *
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn

writer = SummaryWriter(os.path.join(RUNS_PATH, EXPERIMENT_NAME))

if __name__ == '__main__':
    training_data = TIDataset()
    test_data = TIDataset("eval")
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE,
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE,
                                 shuffle=True)
    print(
        f"Starting training with {train_dataloader.__len__() * BATCH_SIZE} "
        f"training samples and {test_dataloader.__len__() * BATCH_SIZE} "
        f"test samples")
    audionet = VIT(IMAGE_SIZE).to(DEVICE)
    start_epoch = 1
    if RESUME_TRAIN:
        audionet = torch.load(RESUME_TRAIN_PATH)
    # TODO add scheduler
    optimizer = torch.optim.Adam(audionet.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(start_epoch, EPOCHS):
        print(f"\n---------------------- Epoch {epoch} ----------------------")
        # validation
        audionet.eval()
        total_corrects = 0
        total_labels = 0

        with torch.no_grad():
            avg_loss = 0
            for audios, labels in tqdm(test_dataloader):
                audios, labels = audios.to(DEVICE), labels.to(DEVICE)
                # print(audios.shape, labels.shape)
                y_pred = audionet(audios)
                # print(y_pred.shape, labels.shape)
                loss = criterion(y_pred, labels)
                # print(loss.item())
                avg_loss += loss.item()
                # print(torch.argmax(y_pred, dim=1), labels, torch.argmax(y_pred, dim=1) == labels, torch.sum((torch.argmax(y_pred, dim=1) == labels).to(torch.long)))
                total_corrects += torch.sum((torch.argmax(y_pred, dim=1) == labels).to(torch.long))
                total_labels += labels.shape[0]

            test_loss = avg_loss / len(test_dataloader)
            writer.add_scalar('Loss/test', test_loss, epoch)
            print("accuracy",  total_corrects / total_labels * 100)
            writer.add_scalar("Accuracy/test",
                              total_corrects / total_labels * 100, epoch)

        audionet.train()
        avg_loss = 0
        with torch.autograd.set_detect_anomaly(True):
            for audios, labels in tqdm(train_dataloader):
                audios, labels = audios.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                y_pred = audionet(audios)
                loss = criterion(y_pred, labels)
                avg_loss += loss.item()
                loss.backward()
                optimizer.step()
        train_loss = avg_loss / len(train_dataloader)
        writer.add_scalar('Loss/train', train_loss, epoch)

        if epoch % EPOCH_AFTER_SAVE_MODEL == 0:
            if not os.path.exists(MODEL_PATH):
                os.makedirs(MODEL_PATH)
            # logic to save best model
            models = os.listdir(MODEL_PATH)
            models_suffixes = [i.split(".pth")[0].split("_")[0] for i in models]
            if "best" in models_suffixes:
                best_model = models[models_suffixes.index("best")]
                prev_loss = float(best_model.split(".pth")[0].split("_")[-1])
                if prev_loss > test_loss:
                    os.remove(os.path.join(MODEL_PATH, best_model))
                    torch.save(audionet.state_dict(),
                               os.path.join(MODEL_PATH,
                                            f"best_{test_loss}.pth"))
            else:
                torch.save(audionet.state_dict(),
                           os.path.join(MODEL_PATH, f"best_{test_loss}.pth"))
            # logic to keep no of models less than MODELS_TO_KEEP
            models_available = sorted(os.listdir(MODEL_PATH),
                                      key=lambda x: float(
                                          x.split(".pth")[0].split("_")[-1]))
            if len(models_available) > MODELS_TO_KEEP:
                # 0 element is "best_model.pth" so keep it .
                os.remove(os.path.join(MODEL_PATH, models_available[1]))
            torch.save(audionet.state_dict(),
                       os.path.join(MODEL_PATH, f"epoch_{epoch}.pth"))
