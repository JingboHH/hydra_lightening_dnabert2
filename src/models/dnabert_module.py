from lightning.pytorch import LightningModule
from transformers import AutoModelForSequenceClassification, AdamW
import torch.nn.functional as F


class DNABERTModule(LightningModule):
    def __init__(self, model_name_or_path, learning_rate, num_labels, **kwargs):
        """
        Initialize the DNABERT2 model.
        :param model_name_or_path: Path to the pretrained DNABERT2 model.
        :param learning_rate: Learning rate for the optimizer.
        :param num_labels: Number of output labels for classification.
        :param kwargs: Additional arguments.
        """
        super().__init__()
        self.save_hyperparameters()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            trust_remote_code=True,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Forward pass through the DNABERT2 model.
        :param input_ids: Tokenized input sequences.
        :param attention_mask: Attention mask for the sequences.
        :return: Model output.
        """
        print(f"Input IDs Type: {type(input_ids)}, Shape: {input_ids.shape}")
        print(
            f"Attention Mask Type: {type(attention_mask)}, Shape: {attention_mask.shape}"
        )
        print(f"Labels Type: {type(labels)}, Value: {labels}")

        return self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.
        :param batch: Batch of data.
        :param batch_idx: Index of the batch.
        :return: Training loss.
        """
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.
        :param batch: Batch of data.
        :param batch_idx: Index of the batch.
        :return: Validation loss.
        """
        outputs = self(**batch)
        val_loss = outputs.loss
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        """
        Perform a single test step.
        :param batch: Batch of data.
        :param batch_idx: Index of the batch.
        :return: Test loss and metrics.
        """
        outputs = self(**batch)
        test_loss = outputs.loss
        self.log("test_loss", test_loss, on_step=False, on_epoch=True, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        """
        Configure the optimizer for training.
        :return: Optimizer.
        """
        return AdamW(self.parameters(), lr=self.hparams.learning_rate)
