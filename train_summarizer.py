# Import necessary libraries
pip install optuna arxiv
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd
import optuna
import arxiv

# Configuration
CONFIG = {
    "data": {
        "max_papers": 150,
        "max_length": 512,
        "summary_length": 250,
        "save_path": "./data",
    },
    "model": {
        "name": "t5-base",
    },
}

# Dataset Class
class ResearchPaperDataset(Dataset):
    def __init__(self, texts, summaries, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.summaries = summaries
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = "summarize: " + str(self.texts[idx])
        summary = str(self.summaries[idx])

        inputs = self.tokenizer(
            text, max_length=self.max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        targets = self.tokenizer(
            summary, max_length=CONFIG["data"]["summary_length"], padding="max_length", truncation=True, return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": targets["input_ids"].squeeze(0),
        }

# Fetch research papers from arXiv
def fetch_arxiv_papers(query, max_results=100):
    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = [{"text": paper.summary.strip(), "abstract": paper.title.strip()} for paper in search.results()]
    return pd.DataFrame(papers)

# Load and prepare data
def prepare_data():
    df = fetch_arxiv_papers("AI", CONFIG["data"]["max_papers"])
    if df.empty:
        raise ValueError("No papers found for the given query.")

    train_texts, val_texts, train_summaries, val_summaries = train_test_split(
        df["text"], df["abstract"], test_size=0.1, random_state=42
    )

    train_texts = train_texts.reset_index(drop=True)
    val_texts = val_texts.reset_index(drop=True)
    train_summaries = train_summaries.reset_index(drop=True)
    val_summaries = val_summaries.reset_index(drop=True)

    return train_texts, val_texts, train_summaries, val_summaries

# Objective function for Optuna
def objective(trial):
    # Define hyperparameters to tune
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 5e-4)
    batch_size = trial.suggest_categorical("batch_size", [2, 4, 8])
    warmup_steps = trial.suggest_int("warmup_steps", 0, 1000)
    num_beams = trial.suggest_int("num_beams", 2, 8)

    # Model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"])
    model = T5ForConditionalGeneration.from_pretrained(CONFIG["model"]["name"])
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset preparation
    train_dataset = ResearchPaperDataset(train_texts, train_summaries, tokenizer, max_length=CONFIG["data"]["max_length"])
    val_dataset = ResearchPaperDataset(val_texts, val_summaries, tokenizer, max_length=CONFIG["data"]["max_length"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./optuna_results",
        num_train_epochs=8,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=8 // batch_size,  # To simulate larger batch sizes
        evaluation_strategy="epoch",
        save_total_limit=1,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train and evaluate
    trainer.train()
    metrics = trainer.evaluate()
    val_loss = metrics["eval_loss"]

    # Return validation loss (lower is better)
    return val_loss

# Create and run Optuna study
if __name__ == "__main__":
    # Prepare data
    train_texts, val_texts, train_summaries, val_summaries = prepare_data()

    # Run Optuna
    study = optuna.create_study(direction="minimize")  # Minimize validation loss
    study.optimize(objective, n_trials=10)  # Run for 10 trials

    # Best trial results
    print("Best trial:")
    print(study.best_trial.params)

    # Save the best model after Optuna optimization
    best_trial = study.best_trial
    best_model = T5ForConditionalGeneration.from_pretrained(CONFIG["model"]["name"])
    best_model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load the best hyperparameters from the best trial
    learning_rate = best_trial.params["learning_rate"]
    batch_size = best_trial.params["batch_size"]
    warmup_steps = best_trial.params["warmup_steps"]
    num_beams = best_trial.params["num_beams"]

    # Dataset preparation for the best model
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model"]["name"])
    train_dataset = ResearchPaperDataset(train_texts, train_summaries, tokenizer, max_length=CONFIG["data"]["max_length"])
    val_dataset = ResearchPaperDataset(val_texts, val_summaries, tokenizer, max_length=CONFIG["data"]["max_length"])

    # Set the training arguments for the best model
    training_args = TrainingArguments(
        output_dir="./best_model_output",
        num_train_epochs=9,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=16 // batch_size,
        evaluation_strategy="epoch",
        save_total_limit=1,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        logging_dir="./logs",
        logging_steps=10,
        fp16=True,
    )

    # Trainer for the best model
    trainer = Trainer(
        model=best_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # Train the best model
    trainer.train()

    # Save the best model and tokenizer
    best_model.save_pretrained("./best_model")
    tokenizer.save_pretrained("./best_model")

    print("Best model saved successfully at './best_model'.")
