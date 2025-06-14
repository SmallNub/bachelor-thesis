import numpy as np
import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    TrainerCallback,
)

from metrics import compute_metrics
from config import MODELS_DIR


# Label smoothing for CrossEntropyLoss
LABEL_SMOOTHING = 0.05


def load_tokenizer(model_name):
    """Load the tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=MODELS_DIR,
        use_fast=True,
    )
    return tokenizer


def load_model(model_name, tokenizer, use_cot):
    """Load the quantized model."""
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model
    model = WeightedLossT5.from_pretrained(
        model_name,
        cache_dir=MODELS_DIR,
        torch_dtype=torch.bfloat16,  # Incompatible with deepspeed?
        local_files_only=False,  # Change for first time downloads
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map="auto",  # Incompatible with deepspeed
    )
    model.tokenizer = tokenizer
    model.use_cot = use_cot
    return model


class WeightedLossT5(T5ForConditionalGeneration):
    """A T5 model with built-in weighted loss (or penalty)."""
    use_cot = False
    tokenizer = None
    current_epoch = 0

    def set_epoch(self, epoch: int):
        """Set the current epoch."""
        self.current_epoch = epoch

    def decode_tokens(self, encoded: np.ndarray) -> list[str]:
        """Replace PyTorch pad token id with tokenizer token id and decode it."""
        encoded = np.where(encoded != -100, encoded, self.tokenizer.pad_token_id)
        decoded = self.tokenizer.batch_decode(encoded, skip_special_tokens=True)
        return decoded

    # Will be called twice during one prediction step, first by generate then forward
    # Generate will not contain labels thus no loss computation
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        outputs = super().forward(
            input_ids,
            attention_mask,
            decoder_input_ids,
            decoder_attention_mask,
            head_mask,
            decoder_head_mask,
            cross_attn_head_mask,
            encoder_outputs,
            past_key_values,
            inputs_embeds,
            decoder_inputs_embeds,
            labels,
            use_cache,
            output_attentions,
            output_hidden_states,
            return_dict,
            cache_position,
        )
        # Cannot compute penalties without labels
        if labels is None:
            return outputs

        is_output_dict = isinstance(outputs, dict)

        logits = outputs["logits"] if is_output_dict else outputs[1]
        loss = outputs["loss"] if is_output_dict else outputs[0]

        # Shifting is required for causal LM
        # Simple workaround to save compute, not the same as model.generate
        encoded_preds = logits.argmax(dim=-1).detach().cpu().numpy()
        encoded_labels = labels.detach().cpu().numpy()
        decoded_preds = self.decode_tokens(encoded_preds)
        decoded_labels = self.decode_tokens(encoded_labels)

        _, penalties = compute_metrics(
            decoded_preds,
            decoded_labels,
            use_cot=self.use_cot,
            current_epoch=self.current_epoch
        )
        penalties = torch.tensor(penalties, dtype=torch.float32, device=logits.device)

        B, T, C = logits.shape
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none", label_smoothing=LABEL_SMOOTHING)

        # Flatten for loss computation
        loss = loss_fct(logits.view(-1, C), labels.view(-1))  # shape: (B * T,)

        # Reshape back to (B, T)
        loss = loss.view(B, T)  # shape: (B, T)

        # Mask for ignored tokens
        mask = (labels != -100).float()  # shape: (B, T)

        # Apply per-sample penalties
        loss = (loss * mask * penalties[:, None]).sum() / mask.sum()

        # Overwrite the loss
        if is_output_dict:
            outputs["loss"] = loss
        else:
            outputs[0] = loss

        return outputs


class CustomTrainer(Seq2SeqTrainer):
    """Custom trainer for handling custom metrics."""
    debug = False
    use_cot = False
    metrics_input_data = None
    current_split = "eval"

    def decode_tokens(self, encoded: np.ndarray) -> list[str]:
        """Replace PyTorch pad token id with tokenizer token id and decode it."""
        encoded = np.where(encoded != -100, encoded, self.processing_class.pad_token_id)
        decoded = self.processing_class.batch_decode(encoded, skip_special_tokens=True)
        return decoded

    def _compute_metrics(self, eval_preds, split: str = None):
        """Compute custom metrics."""
        if split is None:
            split = self.current_split

        encoded_preds, encoded_labels = eval_preds
        preds = self.decode_tokens(encoded_preds)
        labels = self.decode_tokens(encoded_labels)

        if self.debug:
            print("Split:", split)
            print("Preds:", preds)
            print("Labels:", labels)

        metrics, _ = compute_metrics(
            preds,
            labels,
            use_cot=self.use_cot,
            current_epoch=int(self.state.epoch)
        )

        return metrics

    # Label smoothing is not used if the training args do not use label smoothing
    # It will use the model to output the loss instead of trainer

    # Dirty fix, modifies default loss using predictions
    # Called during training, will fail outside training
    # Trainer does call this during eval and test,
    # but Seq2SeqTrainer does not - very weird
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss, outputs = super().compute_loss(model, inputs, True, num_items_in_batch)
        return (loss, outputs) if return_outputs else loss


class EpochTrackerCallback(TrainerCallback):
    """Sets the current epoch inside the model."""
    def on_epoch_begin(self, args, state, control, model, **kwargs):
        if hasattr(model, 'set_epoch'):
            model.set_epoch(int(state.epoch))
