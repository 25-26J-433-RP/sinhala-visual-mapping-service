{"text": "මුහුදු වෙරළ අප රටට ස්වභාවධර්මයෙන් ලැබුණු වටිනා දායාදයකි. කොරල් පර සහ කඩොලාන පද්ධති වෙරළ ඛාදනය වැළැක්වීමට මහත් පිටුවහලක් වේ. එහෙත් මිනිස් ක්‍රියාකාරකම් නිසා මෙම වෙරළ තීරය දූෂණය වීම කණගාටුදායකය. ප්ලාස්ටික් සහ පොලිතින් මුහුදට දැමීමෙන් සාගර ජීවීන් විශාල අනතුරකට ලක් වේ. එබැවින් අපේ වෙරළ පද්ධතිය රැක ගැනීම සැමගේ යුතුකමකි.", "labels": [[0, 11, "multiword-concept"], [46, 54, "multiword-concept"], [58, 76, "multiword-concept"], [77, 90, "multiword-concept"], [125, 137, "multiword-concept"], [166, 175, "concept"], [180, 188, "concept"], [201, 212, "multiword-concept"], [246, 260, "multiword-concept"]]}
{"text": "මගේ අනාගත බලාපොරොත්තුව වෛද්‍යවරයකු වීමයි. රෝගීන්ගේ දුක් වේදනා හඳුනාගෙන ඔවුන්ට ප්‍රතිකාර කිරීම උතුම් සේවාවකි. මේ සඳහා මා දැන් සිටම විද්‍යා විෂයන් හොඳින් හැදෑරිය යුතුය. රෝහල් පද්ධතිය තුළ කාර්යක්ෂමව වැඩ කිරීමට මම දැඩි උත්සාහයක් දරන්නෙමි. දුප්පත් ජනතාවට නොමිලේ සේවය කිරීම මගේ පරම අභිලාෂයයි.", "labels": [[5, 26, "multiword-concept"], [27, 40, "concept"], [83, 93, "concept"], [118, 132, "multiword-concept"], [153, 169, "multiword-concept"], [231, 246, "multiword-concept"]]}
{"text": "මහත්මා ගාන්ධිතුමා ලෝකයටම අවිහිංසාව පිළිබඳ ආදර්ශයක් දුන් ශ්‍රේෂ්ඨ නායකයෙකි. ඔහු ඉන්දියාවේ නිදහස් සටන මෙහෙයවූයේ සාමකාමී ආකාරයටය. සත්‍යග්‍රහ ව්‍යාපාරය මගින් ඔහු අධිරාජ්‍යවාදී පාලනයට එරෙහිව සටන් කළේය. මිනිස් අයිතිවාසිකම් සහ සමානාත්මතාවය වෙනුවෙන් පෙනී සිටි ඔහුගේ චරිතය අදටත් අපට වටිනා පාඩම් කියා දෙයි.", "labels": [[0, 16, "multiword-concept"], [26, 36, "concept"], [74, 94, "multiword-concept"], [115, 136, "multiword-concept"], [145, 168, "multiword-concept"], [189, 210, "multiword-concept"], [215, 233, "concept"]]}
{"text": "ස්වභාවික ආපදා හදිසියේ පැමිණිය හැකි බැවින් අප ඒ සඳහා සූදානම් විය යුතුය. ගංවතුර, නායයෑම් සහ සුළි සුළං නිසා දේපළ මෙන්ම ජීවිත හානි ද සිදු වේ. ආපදා කළමනාකරණ මධ්‍යස්ථානය මගින් ලබා දෙන උපදෙස් පිළිපැදීම ඉතා වැදගත් වේ. හදිසි අවස්ථාවකදී එකිනෙකාට උදව් කර ගැනීමෙන් මෙවැනි විපත්වල බලපෑම අවම කර ගත හැකිය.", "labels": [[0, 15, "multiword-concept"], [73, 80, "concept"], [82, 89, "concept"], [94, 103, "multiword-concept"], [122, 134, "multiword-concept"], [141, 171, "multiword-concept"], [216, 231, "multiword-concept"]]}
"""
Train a Sinhala concept extractor (token/phrase classification) using HuggingFace Transformers.
Labels: concept, multiword-concept, non-concept.
"""
import os
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

MODEL_NAME = "xlm-roberta-base"  # Good multilingual base
LABEL_LIST = ["O", "B-concept", "B-multiword-concept"]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}

# 1. Load your annotated data (JSONL: {"text":..., "labels":[[start,end,label],...]})
def load_annotated_data(path):
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            text = item["text"]
            tags = ["O"] * len(text)
            for start, end, label in item["labels"]:
                if label == "concept":
                    tags[start] = "B-concept"
                elif label == "multiword-concept":
                    tags[start] = "B-multiword-concept"
            samples.append({"text": text, "tags": tags})
    return samples

def tokenize_and_align_labels(examples, tokenizer):
    tokenized = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=128, return_offsets_mapping=True
    )
    labels = []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        word_labels = examples["tags"][i]
        label_ids = []
        for offset in offsets:
            if offset[0] == offset[1]:
                label_ids.append(-100)
            else:
                idx = offset[0]
                label_ids.append(LABEL2ID.get(word_labels[idx], 0) if idx < len(word_labels) else 0)
        labels.append(label_ids)
    tokenized["labels"] = labels
    return tokenized

def main():
    # Path to your annotated JSONL file
    train_path = "concept_train.jsonl"
    val_path = "concept_val.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_samples = load_annotated_data(train_path)
    val_samples = load_annotated_data(val_path)
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)
    train_dataset = train_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_and_align_labels(x, tokenizer), batched=True)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_LIST), id2label=ID2LABEL, label2id=LABEL2ID
    )
    args = TrainingArguments(
        output_dir="concept-extractor-model",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=20,
        save_total_limit=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained("concept-extractor-model")
    tokenizer.save_pretrained("concept-extractor-model")

if __name__ == "__main__":
    main()
