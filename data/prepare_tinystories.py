# tinystories dataset
# modified from NanoGPT's openwebtext script

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from datasets import load_dataset  # huggingface datasets
from tqdm import tqdm

from tart import vocab

num_proc = 8
root_dir = os.path.join(os.path.dirname(__file__), "tinystories")
os.makedirs(root_dir, exist_ok=True)

if __name__ == "__main__":
    dataset = load_dataset("roneneldan/TinyStories", num_proc=num_proc)

    # this results in:
    # >>> split_dataset
    # DatasetDict({
    #     train: Dataset({
    #         features: ['text'],
    #         num_rows: 8009762
    #     })
    #     val: Dataset({
    #         features: ['text'],
    #         num_rows: 4007
    #     })
    # })

    def process(example):
        tokens = vocab.encode_doc(example["text"])
        out = {"tokens": tokens, "len": len(tokens)}
        return out

    # tokenize the dataset
    processed = dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in processed.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        print(f"split {split} has {arr_len:,} tokens")
        filename = os.path.join(root_dir, f"{split}.bin")
        dtype = np.uint16
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Batch together samples for faster write
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["tokens"]).astype(dtype)
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
