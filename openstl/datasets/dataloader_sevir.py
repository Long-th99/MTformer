#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader


class SevirNpyDataset(Dataset):
    def __init__(self, data_root, split='train', image_size=64,
                 pre_seq_length=10, aft_seq_length=10, step=1,
                 channels=1, train_ratio=0.8, use_augment=False):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length = pre_seq_length + aft_seq_length
        self.step = step
        self.channels = channels
        self.use_augment = use_augment
        self.train_ratio = train_ratio
        
        npy_path = os.path.join(data_root, 'sevir.npy')
        if not os.path.isfile(npy_path):
            raise FileNotFoundError(npy_path)
        self.data = np.load(npy_path).astype(np.float32)
        self.B, self.T, self.C, self.H, self.W = self.data.shape
        if self.C != self.channels:
            raise ValueError(f'channels mismatch: expected {self.channels}, got {self.C}')

        self.indices = self._build_indices()
        self.mean = None
        self.std = None
        print(f'{split} set: {len(self.indices)} windows')

    def _build_indices(self):
        B = self.B
        split_idx = int(B * self.train_ratio)
        if self.split == 'train':
            b_range = range(0, split_idx)
        else:
            b_range = range(split_idx, B)

        max_start = self.T - self.seq_length + 1
        indices = []
        for b in b_range:
            for start in range(0, max_start, self.step):
                indices.append((b, start))
        return indices

    def _normalize(self, x):
        xmin, xmax = x.min(), x.max()
        return (x - xmin) / (xmax - xmin + 1e-8)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        b, start = self.indices[idx]
        end = start + self.seq_length
        seq = self.data[b, start:end, :, :, :]

        resized = np.zeros((self.seq_length, self.channels, self.image_size, self.image_size), dtype=np.float32)
        for t in range(self.seq_length):
            for c in range(self.channels):
                frame = seq[t, c, :, :]
                if (self.H, self.W) != (self.image_size, self.image_size):
                    frame = cv2.resize(frame, (self.image_size, self.image_size),
                                       interpolation=cv2.INTER_LINEAR)
                resized[t, c, :, :] = self._normalize(frame)

        data = torch.from_numpy(resized[:self.pre_seq_length]).float()
        labels = torch.from_numpy(resized[self.pre_seq_length:]).float()
        return data, labels


def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, in_shape=None,
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):
    if in_shape is None:
        in_shape = [10, 1, 64, 64]
    image_size = in_shape[-1]
    channels = in_shape[1] if len(in_shape) > 1 else 1

    train_set = SevirNpyDataset(data_root, split='train', image_size=image_size,
                                pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                                step=1, channels=channels, use_augment=use_augment)
    test_set = SevirNpyDataset(data_root, split='test', image_size=image_size,
                               pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                               step=1, channels=channels, use_augment=False)

    dataloader_train = create_loader(train_set, batch_size=batch_size, shuffle=True,
                                     is_training=True, pin_memory=True, drop_last=True,
                                     num_workers=num_workers, distributed=distributed,
                                     use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set, batch_size=val_batch_size, shuffle=False,
                                    is_training=False, pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers, distributed=distributed,
                                    use_prefetcher=use_prefetcher)
    return dataloader_train, dataloader_test, dataloader_test


if __name__ == '__main__':
    dl_train, _, dl_test = load_data(batch_size=4, val_batch_size=4,
                                     data_root='data/',
                                     num_workers=0,
                                     pre_seq_length=14, aft_seq_length=10,
                                     in_shape=[24, 1, 64, 64])
    for x, y in dl_train:
        print("Input :", x.shape)
        print("Label :", y.shape)
        break