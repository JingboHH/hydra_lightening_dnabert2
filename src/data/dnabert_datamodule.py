from lightning.pytorch import LightningDataModule
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd  # For loading CSV files
import torch
import os


class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        sequence = item["sequence"]  # Assuming 'sequence' column exists
        label = item["label"]  # Assuming 'label' column exists

        # Tokenize the DNA sequence
        tokenized = self.tokenizer(
            sequence, max_length=self.max_length, truncation=True, padding="max_length"
        )

        # Ensure the outputs are converted to tensors
        return {
            "input_ids": torch.tensor(tokenized["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(
                tokenized["attention_mask"], dtype=torch.long
            ),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class DNABERTDataModule(LightningDataModule):
    def __init__(self, data_path, batch_size, max_length, model_name_or_path):
        super().__init__()
        self.data_path = data_path  # Base path to the dataset
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def setup(self, stage=None):
        # Correctly construct paths for train, dev, and test CSV files
        train_file = os.path.join(self.data_path, "train.csv")
        dev_file = os.path.join(self.data_path, "dev.csv")
        test_file = os.path.join(self.data_path, "test.csv")

        # Verify that files exist
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file not found at {train_file}")
        if not os.path.exists(dev_file):
            raise FileNotFoundError(f"Development file not found at {dev_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Test file not found at {test_file}")

        # Load datasets
        print("Loading datasets...")
        self.train_data = CustomDataset(
            pd.read_csv(train_file), self.tokenizer, self.max_length
        )
        self.val_data = CustomDataset(
            pd.read_csv(dev_file), self.tokenizer, self.max_length
        )
        self.test_data = CustomDataset(
            pd.read_csv(test_file), self.tokenizer, self.max_length
        )

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size)
