import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from openstl.datasets.utils import create_loader

class NavierStokesT20Dataset(Dataset):
    def __init__(self, data_root, split='train', image_size=64,
                 pre_seq_length=10, aft_seq_length=10, step=1, use_augment=False, channels=1, train_ratio=0.8, eval_mode=False):
        super(NavierStokesT20Dataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        self.pre_seq_length = pre_seq_length
        self.aft_seq_length = aft_seq_length
        self.seq_length = pre_seq_length + aft_seq_length
        self.step = step
        self.use_augment = use_augment
        self.channels = channels
        self.train_ratio = train_ratio
        self.eval_mode = eval_mode
        self.input_shape = (self.seq_length, self.channels, self.image_size, self.image_size)
        self.file_list = self._build_file_list()
        self.mean = None
        self.std = None
        self.data_cache = {}
        print(f"{self.split} sequences: {len(self.file_list)}")

    def _build_file_list(self):
        npy_files = [f for f in os.listdir(self.data_root) if f.endswith('.npy')]
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {self.data_root}. Please add data files.")
        
        all_sequences = []
        for npy_file in npy_files:
            npy_path = os.path.join(self.data_root, npy_file)
            try:
                data = np.load(npy_path)
                print(f"Loaded {npy_file} shape: {data.shape}, dtype: {data.dtype}")
            
                if len(data.shape) == 4:
                    T, H, W, N = data.shape
                    data = np.transpose(data, (3, 0, 1, 2))
                    data = data[..., np.newaxis]
                    data = np.transpose(data, (0, 1, 4, 2, 3))
                elif len(data.shape) == 3:
                    T, H, W = data.shape
                    data = data[np.newaxis, ...]
                    data = data[..., np.newaxis]
                    data = np.transpose(data, (0, 1, 4, 2, 3))
                elif len(data.shape) != 5:
                    print(f"Warning: Unexpected shape {data.shape} in {npy_file}, skipping.")
                    continue
            
                N, T, C, H, W = data.shape
                print(f"Adjusted shape: (N={N}, T={T}, C={C}, H={H}, W={W})")
            
                if C != self.channels:
                    print(f"Warning: Channels mismatch in {npy_file}: expected {self.channels}, got {C}")
                    continue
            
                seq_step = self.step
                if self.eval_mode:
                    start_step = self.seq_length
                else:
                    start_step = 1
                
                max_start = max(0, T - (self.seq_length - 1) * seq_step)
                
                num_sequences = 0
                for n in range(N):
                    start_idx = 0
                    while start_idx < max_start:
                        all_sequences.append(f"{npy_file},{n},{start_idx}")
                        num_sequences += 1
                        start_idx += start_step
                
                print(f"From {npy_file}: {num_sequences} sequences generated (eval_mode={self.eval_mode})")
            
            except Exception as e:
                print(f"Error loading {npy_file}: {e}")
                continue
        
        if not all_sequences:
            raise ValueError("No valid sequences found! Check .npy shapes.")
        
        random.shuffle(all_sequences)
        split_idx = int(len(all_sequences) * self.train_ratio)
        if self.split == 'train':
            return all_sequences[:split_idx]
        else:
            return all_sequences[split_idx:]

    def _load_npy(self, npy_path):
        if npy_path not in self.data_cache:
            data = np.load(npy_path)
            
            if len(data.shape) == 4:
                T, H, W, N = data.shape
                data = np.transpose(data, (3, 0, 1, 2))
                data = data[..., np.newaxis]
                data = np.transpose(data, (0, 1, 4, 2, 3))
            elif len(data.shape) == 3:
                T, H, W = data.shape
                data = data[np.newaxis, ...]
                data = data[..., np.newaxis]
                data = np.transpose(data, (0, 1, 4, 2, 3))
            elif len(data.shape) != 5:
                raise ValueError(f"Unexpected shape {data.shape} in {npy_path}")
            
            self.data_cache[npy_path] = data.astype(np.float32)
        return self.data_cache[npy_path]

    def _augment_seq(self, frames, h, w):
        length = len(frames)
        x = np.random.randint(0, frames[0].shape[0] - h + 1)
        y = np.random.randint(0, frames[0].shape[1] - w + 1)
        for i in range(length):
            frames[i] = frames[i][x:x+h, y:y+w, :]
        frames = [torch.from_numpy(f.copy()).float() for f in frames]
        return frames

    def _to_tensor(self, frames):
        return [torch.from_numpy(f.copy()).float() for f in frames]

    def _normalize(self, frames):
        frames = np.array(frames)
        frames = (frames - frames.min()) / (frames.max() - frames.min() + 1e-8)
        return frames

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        item_list = self.file_list[idx].split(',')
        npy_file = item_list[0].strip()
        sample_n = int(item_list[1].strip())
        begin = int(item_list[2].strip())
        npy_path = os.path.join(self.data_root, npy_file)
        full_data = self._load_npy(npy_path)
        N, T, C, H, W = full_data.shape
        if C != self.channels:
            raise ValueError(f"Channels mismatch: expected {self.channels}, got {C}")
        sample_data = full_data[sample_n]
        sample_data = np.transpose(sample_data, (0, 2, 3, 1))
        
        end = begin + self.seq_length * self.step
        seq_indices = list(range(begin, min(end, T), self.step))
        if len(seq_indices) < self.seq_length:
            last_idx = seq_indices[-1]
            while len(seq_indices) < self.seq_length:
                seq_indices.append(last_idx)
        seq_indices = seq_indices[:self.seq_length]
        frames = sample_data[seq_indices]
        
        if H != self.image_size or W != self.image_size:
            resized_frames = []
            for frame in frames:
                frame = frame.astype(np.float32)
                if self.channels == 1:
                    resized = cv2.resize(frame.squeeze(-1), (self.image_size, self.image_size),
                                         interpolation=cv2.INTER_CUBIC)
                    resized = resized[..., np.newaxis]
                else:
                    resized = np.zeros((self.image_size, self.image_size, self.channels))
                    for c in range(self.channels):
                        resized[:,:,c] = cv2.resize(frame[:,:,c], (self.image_size, self.image_size),
                                                     interpolation=cv2.INTER_CUBIC)
                resized_frames.append(resized)
            frames = np.array(resized_frames)
            
        frames = self._normalize(frames)
        
        if self.use_augment:
            frames = self._augment_seq(frames, h=self.image_size, w=self.image_size)
        else:
            frames = self._to_tensor(frames)
            
        img_seq = torch.stack(frames, 0).permute(0, 3, 1, 2)
        data = img_seq[:self.pre_seq_length, ...]
        labels = img_seq[self.pre_seq_length:, ...]
        return data, labels

def load_data(batch_size, val_batch_size, data_root, num_workers=4,
              pre_seq_length=10, aft_seq_length=10, in_shape=[10, 1, 64, 64],
              distributed=False, use_augment=False, use_prefetcher=False, drop_last=False):
    data_root = os.path.join(data_root, 'navier_stokes_t20')
    image_size = in_shape[-1] if len(in_shape) >= 3 else 64
    channels = in_shape[1] if len(in_shape) > 1 else 1
    train_set = NavierStokesT20Dataset(data_root, split='train', image_size=image_size,
                                       pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                                       step=1, use_augment=use_augment, channels=channels, eval_mode=False)
    test_set = NavierStokesT20Dataset(data_root, split='test', image_size=image_size,
                                      pre_seq_length=pre_seq_length, aft_seq_length=aft_seq_length,
                                      step=1, use_augment=False, channels=channels, eval_mode=True)
    dataloader_train = create_loader(train_set,
                                     batch_size=batch_size,
                                     shuffle=True, is_training=True,
                                     pin_memory=True, drop_last=True,
                                     num_workers=num_workers,
                                     distributed=distributed, use_prefetcher=use_prefetcher)
    dataloader_test = create_loader(test_set,
                                    batch_size=val_batch_size,
                                    shuffle=False, is_training=False,
                                    pin_memory=True, drop_last=drop_last,
                                    num_workers=num_workers,
                                    distributed=distributed, use_prefetcher=use_prefetcher)
    return dataloader_train, dataloader_test, dataloader_test

if __name__ == '__main__':
    dataloader_train, _, dataloader_test = load_data(
        batch_size=16, val_batch_size=16, data_root='data/',
        num_workers=0,
        pre_seq_length=10, aft_seq_length=10,
        use_prefetcher=False, distributed=False
    )
    print(f"Train loader len: {len(dataloader_train)}, Test: {len(dataloader_test)}")
    for item in dataloader_train:
        print(f"Input shape: {item[0].shape}, Label shape: {item[1].shape}")
        break