import os

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import librosa
import numpy as np
import pandas as pd

NUM_WORKERS = os.cpu_count()

def load_signal(file_path: str,
                duration: int,
                sample_rate: int):
    y, sr = librosa.load(file_path, duration=duration, sr=sample_rate)

    signal = np.zeros((int(sr*3,)))
    signal[:len(y)] = y

    return signal

def load_mel_spectrogram(file_path: str,
                         duration: int,
                         sample_rate: int, 
                         n_fft: int,
                         win_length:int ,
                         window: str,
                         hop_length: int,
                         n_mels: int):
    fmax = sample_rate / 2
    audio = load_signal(file_path, duration, sample_rate)
    
    mel_specgram = librosa.feature.melspectrogram(y=audio,
                                                  sr=sample_rate,
                                                  n_fft=n_fft,
                                                  win_length=win_length,
                                                  window=window,
                                                  hop_length=hop_length,
                                                  n_mels=n_mels,
                                                  fmax=fmax)

    mel_specgram_df = librosa.power_to_db(mel_specgram, ref=np.max)
    return mel_specgram_df

def split_into_chunks(mel_spectrogram, win_size: int, stride: int):
    t = mel_spectrogram.shape[1]
    num_of_chunks = t // stride
    chunks = []

    for i in range(num_of_chunks):
        chunk = mel_spectrogram[:, i*stride:i*stride+win_size]
        if chunk.shape[1] == win_size:
            chunks.append(chunk)

    return np.stack(chunks, axis=0)

class MelSpectogramsDataset(Dataset):
    def __init__(self,
                 df: pd.DataFrame,
                 duration=3,
                 sample_rate=44100, 
                 n_fft=2048,
                 win_length=512,
                 window="hamming",
                 hop_length=512,
                 n_mels=128,
                 win_size=128,
                 stride=64,
                 transform=None):
        self.df = df
        self.duration=duration
        self.sample_rate=sample_rate
        self.n_fft=n_fft
        self.win_length=win_length
        self.window=window
        self.hop_length=hop_length
        self.n_mels=n_mels
        self.win_size = win_size
        self.stride = stride
        self.transform = transform
        self.classes = self.df["Emotion"].unique()
        self.classes_labels = {class_name: idx for idx, class_name in enumerate(self.classes)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        file_path = self.df.iloc[idx]["Path"]
        
        emotion = self.df.iloc[idx]["Emotion"]
        emotion_label = self.classes_labels[emotion]
        
    
        mel_spectrogram = load_mel_spectrogram(file_path,
                                               self.duration,
                                               self.sample_rate,
                                               self.n_fft,
                                               self.win_length,
                                               self.window,
                                               self.hop_length,
                                               self.n_mels)
        
        mel_chunks = split_into_chunks(mel_spectrogram,
                                       self.win_size,
                                       self.stride)
        
        mel_chunks = torch.tensor(mel_chunks, dtype=torch.float32).unsqueeze(1)
        
        if self.transform:
            mel_chunks = torch.stack([self.transform(chunk) for chunk in mel_chunks])
        
        return mel_chunks, emotion_label
        

def create_dataloaders(
    train_dataframe: pd.DataFrame, 
    test_dataframe: pd.DataFrame, 
    original_transform: transforms.Compose, 
    augment_transform: transforms.Compose,
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  train_data = ConcatDataset([MelSpectogramsDataset(train_dataframe, transform=augment_transform),
                             MelSpectogramsDataset(train_dataframe, transform=original_transform)])
  test_data = MelSpectogramsDataset(test_dataframe, transform=original_transform)

  class_names = test_data.classes

  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names