{"text": "මුහුදු වෙරළ අප රටට ස්වභාවධර්මයෙන් ලැබුණු වටිනා දායාදයකි. කොරල් පර සහ කඩොලාන පද්ධති වෙරළ ඛාදනය වැළැක්වීමට මහත් පිටුවහලක් වේ. එහෙත් මිනිස් ක්‍රියාකාරකම් නිසා මෙම වෙරළ තීරය දූෂණය වීම කණගාටුදායකය. ප්ලාස්ටික් සහ පොලිතින් මුහුදට දැමීමෙන් සාගර ජීවීන් විශාල අනතුරකට ලක් වේ. එබැවින් අපේ වෙරළ පද්ධතිය රැක ගැනීම සැමගේ යුතුකමකි.", "labels": [[0, 11, "multiword-concept"], [46, 54, "multiword-concept"], [58, 76, "multiword-concept"], [77, 90, "multiword-concept"], [125, 137, "multiword-concept"], [166, 175, "concept"], [180, 188, "concept"], [201, 212, "multiword-concept"], [246, 260, "multiword-concept"]]}
{"text": "මගේ අනාගත බලාපොරොත්තුව වෛද්‍යවරයකු වීමයි. රෝගීන්ගේ දුක් වේදනා හඳුනාගෙන ඔවුන්ට ප්‍රතිකාර කිරීම උතුම් සේවාවකි. මේ සඳහා මා දැන් සිටම විද්‍යා විෂයන් හොඳින් හැදෑරිය යුතුය. රෝහල් පද්ධතිය තුළ කාර්යක්ෂමව වැඩ කිරීමට මම දැඩි උත්සාහයක් දරන්නෙමි. දුප්පත් ජනතාවට නොමිලේ සේවය කිරීම මගේ පරම අභිලාෂයයි.", "labels": [[5, 26, "multiword-concept"], [27, 40, "concept"], [83, 93, "concept"], [118, 132, "multiword-concept"], [153, 169, "multiword-concept"], [231, 246, "multiword-concept"]]}
{"text": "මහත්මා ගාන්ධිතුමා ලෝකයටම අවිහිංසාව පිළිබඳ ආදර්ශයක් දුන් ශ්‍රේෂ්ඨ නායකයෙකි. ඔහු ඉන්දියාවේ නිදහස් සටන මෙහෙයවූයේ සාමකාමී ආකාරයටය. සත්‍යග්‍රහ ව්‍යාපාරය මගින් ඔහු අධිරාජ්‍යවාදී පාලනයට එරෙහිව සටන් කළේය. මිනිස් අයිතිවාසිකම් සහ සමානාත්මතාවය වෙනුවෙන් පෙනී සිටි ඔහුගේ චරිතය අදටත් අපට වටිනා පාඩම් කියා දෙයි.", "labels": [[0, 16, "multiword-concept"], [26, 36, "concept"], [74, 94, "multiword-concept"], [115, 136, "multiword-concept"], [145, 168, "multiword-concept"], [189, 210, "multiword-concept"], [215, 233, "concept"]]}
{"text": "ස්වභාවික ආපදා හදිසියේ පැමිණිය හැකි බැවින් අප ඒ සඳහා සූදානම් විය යුතුය. ගංවතුර, නායයෑම් සහ සුළි සුළං නිසා දේපළ මෙන්ම ජීවිත හානි ද සිදු වේ. ආපදා කළමනාකරණ මධ්‍යස්ථානය මගින් ලබා දෙන උපදෙස් පිළිපැදීම ඉතා වැදගත් වේ. හදිසි අවස්ථාවකදී එකිනෙකාට උදව් කර ගැනීමෙන් මෙවැනි විපත්වල බලපෑම අවම කර ගත හැකිය.", "labels": [[0, 15, "multiword-concept"], [73, 80, "concept"], [82, 89, "concept"], [94, 103, "multiword-concept"], [122, 134, "multiword-concept"], [141, 171, "multiword-concept"], [216, 231, "multiword-concept"]]}
"""
Train a Sinhala concept extractor (token/phrase classification) using
HuggingFace Transformers.

Labels
------
  O                    – not a concept
  B-concept            – start of a single-word concept
  B-multiword-concept  – start of a multi-word concept phrase

What's new (hard-negative training)
-------------------------------------
Pass ``--use_hard_negatives`` to activate the ``HardNegativeMiner``.  It
mines O-tagged spans that are semantically close to concept anchors and
produces three kinds of augmented samples:

  1. Upsample   – duplicate sentences containing hard negatives.
  2. Inject     – replace a real concept span with a hard-negative phrase,
                  keeping the O label, so the model learns the boundary.
  3. Concatenate – join a concept-rich sentence with a hard-negative one.

A ``FocalLossTrainer`` replaces the default cross-entropy with focal loss
(γ = 2.0) to down-weight easy examples and focus gradient updates on the
hardest boundary cases.
"""

import argparse
import json
import logging
import os

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "xlm-roberta-base"
LABEL_LIST = ["O", "B-concept", "B-multiword-concept"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


# ---------------------------------------------------------------------------
# Focal loss trainer
# ---------------------------------------------------------------------------

class FocalLossTrainer(Trainer):
    """
    HuggingFace Trainer with focal cross-entropy loss.

    Focal loss (Lin et al., 2017):

    .. math::
        FL(p_t) = -(1 - p_t)^{\\gamma} \\cdot \\log(p_t)

    With γ = 2 the loss contribution from confidently-correct predictions is
    reduced by ~4×, forcing the model to allocate capacity to the borderline
    cases (hard negatives among them).

    Parameters
    ----------
    focal_gamma : float
        Focusing parameter γ.  0 → standard cross-entropy; 2 is the
        recommended default.  Higher values focus more aggressively on
        hard examples at the risk of instability.
    """

    def __init__(self, *args, focal_gamma: float = 2.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma

    def compute_loss(self, model, inputs, return_outputs: bool = False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, num_labels)

        # Flatten sequence dimension
        active_mask = labels.view(-1) != -100        # ignore padding / special tokens
        active_logits = logits.view(-1, model.config.num_labels)[active_mask]
        active_labels = labels.view(-1)[active_mask]

        if active_labels.numel() == 0:
            loss = logits.sum() * 0.0   # degenerate batch — zero gradient
            return (loss, outputs) if return_outputs else loss

        # Standard log-softmax + NLL per token
        log_probs = F.log_softmax(active_logits, dim=-1)           # (N, C)
        ce_per_token = F.nll_loss(log_probs, active_labels, reduction="none")  # (N,)

        # Focal weight: (1 - p_t)^γ
        probs = log_probs.exp()                                     # (N, C)
        pt = probs.gather(1, active_labels.unsqueeze(1)).squeeze(1) # (N,)
        focal_weight = (1.0 - pt).pow(self.focal_gamma)

        loss = (focal_weight * ce_per_token).mean()
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_annotated_data(path: str):
    """
    Load an annotated JSONL file.

    Each line: {"text": "...", "labels": [[start, end, label], ...]}

    Returns a list of {"text": str, "tags": List[str]} dicts where ``tags``
    is a per-character list of label strings (mostly "O").
    """
    samples = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item["text"]
            tags = ["O"] * len(text)
            for start, end, label in item.get("labels", []):
                if label == "concept" and start < len(tags):
                    tags[start] = "B-concept"
                elif label == "multiword-concept" and start < len(tags):
                    tags[start] = "B-multiword-concept"
            samples.append({"text": text, "tags": tags})
    return samples


def tokenize_and_align_labels(examples, tokenizer):
    """Tokenise a batch and align character-level tags to sub-word tokens."""
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_offsets_mapping=True,
    )
    all_label_ids = []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        char_tags = examples["tags"][i]
        label_ids = []
        for offset_start, offset_end in offsets:
            if offset_start == offset_end:
                # Special token (CLS, SEP, PAD)
                label_ids.append(-100)
            else:
                idx = offset_start
                label_ids.append(
                    LABEL2ID.get(char_tags[idx], 0) if idx < len(char_tags) else 0
                )
        all_label_ids.append(label_ids)
    tokenized["labels"] = all_label_ids
    return tokenized


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Sinhala concept extractor with optional hard-negative augmentation."
    )
    parser.add_argument("--train", default="concept_train.jsonl",
                        help="Path to training JSONL file.")
    parser.add_argument("--val", default="concept_val.jsonl",
                        help="Path to validation JSONL file.")
    parser.add_argument("--output_dir", default="concept-extractor-model",
                        help="Directory to save the trained model.")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per-device train/eval batch size.")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate.")
    # Hard-negative options
    parser.add_argument("--use_hard_negatives", action="store_true",
                        help="Enable hard-negative mining and augmentation.")
    parser.add_argument("--hn_sim_threshold", type=float, default=0.55,
                        help="Cosine similarity threshold for hard-negative detection.")
    parser.add_argument("--hn_max", type=int, default=150,
                        help="Maximum number of hard negatives to mine.")
    parser.add_argument("--hn_upsample", type=int, default=1,
                        help="Duplicate count per hard-negative (strategy 1).")
    parser.add_argument("--hn_inject", type=int, default=2,
                        help="Injections per hard-negative (strategy 2).")
    parser.add_argument("--hn_concat", type=int, default=1,
                        help="Concatenations per hard-negative (strategy 3).")
    # Loss options
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss γ. 0 = standard cross-entropy.")
    args = parser.parse_args()

    # ── Build training samples ─────────────────────────────────────────────
    if args.use_hard_negatives:
        logger.info("Hard-negative augmentation ENABLED (sim_threshold=%.2f)",
                    args.hn_sim_threshold)
        from hard_negative_miner import HardNegativeMiner

        # Load originals
        train_samples = load_annotated_data(args.train)
        orig_count = len(train_samples)

        # Mine and augment
        miner = HardNegativeMiner(
            sim_threshold=args.hn_sim_threshold,
            max_hard_negatives=args.hn_max,
            upsample_factor=args.hn_upsample,
            inject_per_hard_neg=args.hn_inject,
            concat_per_hard_neg=args.hn_concat,
        )
        synthetics = miner.mine_from_jsonl(args.train)
        train_samples = train_samples + synthetics
        logger.info(
            "Dataset: %d original + %d synthetic = %d total training samples",
            orig_count, len(synthetics), len(train_samples),
        )
    else:
        logger.info("Hard-negative augmentation DISABLED.")
        train_samples = load_annotated_data(args.train)
        logger.info("Loaded %d training samples.", len(train_samples))

    val_samples = load_annotated_data(args.val)
    logger.info("Loaded %d validation samples.", len(val_samples))

    # ── Tokenise ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = Dataset.from_list(train_samples).map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )
    val_dataset = Dataset.from_list(val_samples).map(
        lambda x: tokenize_and_align_labels(x, tokenizer), batched=True
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Training arguments ─────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=20,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # ── Trainer (focal loss when γ > 0) ───────────────────────────────────
    use_focal = args.focal_gamma > 0.0
    TrainerClass = FocalLossTrainer if use_focal else Trainer
    trainer_kwargs: dict = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    if use_focal:
        trainer_kwargs["focal_gamma"] = args.focal_gamma
        logger.info("Using FocalLossTrainer (γ=%.1f)", args.focal_gamma)
    else:
        logger.info("Using standard cross-entropy loss.")

    trainer = TrainerClass(**trainer_kwargs)

    # ── Train ──────────────────────────────────────────────────────────────
    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()

