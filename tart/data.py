import os

import numpy as np
import torch
import torch.utils.data


class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, path, chunk_size):
        self.chunk_size = chunk_size
        self.data = np.memmap(path, dtype=np.uint16, mode="r")

    @property
    def num_tokens(self):
        return self.data.size

    def __len__(self):
        return self.data.size // (self.chunk_size - 2)

    def __getitem__(self, index):
        index = index * (self.chunk_size - 2)
        # index = index * self.stride
        chunk = self.data[index : index + self.chunk_size]
        return torch.from_numpy(chunk.astype(np.int64))


def create_dataset(root: str, split: str, chunk_size: int):
    path = os.path.join(root, f"{split}.bin")
    return TokenDataset(
        path,
        chunk_size,
    )
