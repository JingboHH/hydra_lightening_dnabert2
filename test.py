from src.data import (
    dnabert_datamodule,
)  # Replace 'your_module' with the actual file/module name
import os

# Initialize the DNABERTDataModule
data_module = dnabert_datamodule(
    data_path=os.path.expanduser("~/data/dnabert/EMP/H3"),  # Path to your dataset
    batch_size=32,  # Set your desired batch size
    max_length=512,  # Max sequence length for tokenizer
    model_name_or_path="zhihan1996/DNABERT-2-117M",  # Pretrained model name
)

# Setup the data
data_module.setup()

# Get the DataLoaders
train_loader = data_module.train_dataloader()
val_loader = data_module.val_dataloader()
test_loader = data_module.test_dataloader()

# Debugging: Check DataLoader lengths
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(val_loader)}")
print(f"Number of test batches: {len(test_loader)}")

# Fetch a batch of data to confirm it's working
for batch in train_loader:
    print("Sample Batch:")
    print(batch)
    break  # Stop after printing the first batch
