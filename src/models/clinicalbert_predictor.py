
"""
Fine-tune ClinicalBERT on MoA summaries (train/val/test CSVs) and report AUROC/AUPR.
"""
import os, json, argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    assert {"text", "label"}.issubset(df.columns), "CSV must contain 'text' and 'label' columns"
    return Dataset.from_pandas(df[["text", "label"]])

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    return {"auroc": roc_auc_score(labels, probs), "aupr": average_precision_score(labels, probs)}

def train_eval(train_csv, val_csv, test_csv, model_id="emilyalsentzer/Bio_ClinicalBERT",
               out_dir="./outputs/models/clinicalbert", epochs=2, lr=2e-5, batch_size=8, max_len=768):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    collator = DataCollatorWithPadding(tok)

    def tokenize(example):
        return tok(example["text"], truncation=True, max_length=max_len)

    ds_train = load_data(train_csv).map(tokenize, batched=True)
    ds_val   = load_data(val_csv).map(tokenize, batched=True)
    ds_test  = load_data(test_csv).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=2)

    args = TrainingArguments(
        output_dir=out_dir,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="aupr",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(ds_test)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics_test.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print("Test metrics:", eval_metrics)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--val_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--model_id", default="emilyalsentzer/Bio_ClinicalBERT")
    ap.add_argument("--out_dir", default="./outputs/models/clinicalbert")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=768)
    args = ap.parse_args()
    train_eval(args.train_csv, args.val_csv, args.test_csv, args.model_id, args.out_dir, args.epochs, args.lr, args.batch_size, args.max_len)
