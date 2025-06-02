import numpy as np
import torch
from transformers import Seq2SeqTrainer

from metrics import extract_docid, compute_match_accuracy


class CustomTrainer(Seq2SeqTrainer):
    debug = False
    use_cot = False
    metrics_input_data = None
    current_split = "eval"
    max_acc_loss_mult = 2.0
    min_acc_loss_mult = 0.5
    _diff_acc_loss_mult = max_acc_loss_mult - min_acc_loss_mult

    def decode(self, encoded: np.ndarray) -> list[str]:
        """Replace PyTorch pad token id with tokenizer token id and decode it."""
        encoded = np.where(encoded != -100, encoded, self.processing_class.pad_token_id)
        decoded = self.processing_class.batch_decode(encoded, skip_special_tokens=True)
        return decoded

    def _compute_metrics(self, eval_preds, split: str = None):
        """Compute metrics."""
        if split is None:
            split = self.current_split

        encoded_preds, encoded_labels = eval_preds
        preds = self.decode(encoded_preds)
        labels = self.decode(encoded_labels)

        if self.use_cot:
            labels = [extract_docid(label) for label in labels]

        if self.debug:
            print("Split:", split)
            print("Preds:", preds)
            print("Labels:", labels)

        accuracy = compute_match_accuracy(preds, labels)

        return {"match_accuracy": accuracy}

    # Label smoothing is not used if the training args do not use label smoothing
    # It will use the model to output the loss instead of trainer

    # Dirty fix, modifies default loss using predictions
    # Called during training, will fail outside training
    # Trainer does call this during eval and test,
    # but Seq2SeqTrainer does not - very weird
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)

        # Shifting is required for causal LM
        # Simple workaround to save compute, not the same as model.generate
        encoded_preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().numpy()
        encoded_labels = inputs["labels"].detach().cpu().numpy()

        metrics = self._compute_metrics((encoded_preds, encoded_labels), "train")
        loss *= self.max_acc_loss_mult - self._diff_acc_loss_mult * metrics["match_accuracy"]

        return (loss, outputs) if return_outputs else loss
