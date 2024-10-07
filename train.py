from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from huggingface_hub import login
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

login(token="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")

# Charger le dataset IMDB
dataset = load_dataset("openai/MMMLU")

# Charger le tokenizer pour BERT
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

# Fonction pour tokenizer les données textuelles
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Appliquer la tokenization au dataset d'entraînement et de test
train_dataset = dataset['train'].map(tokenize_function, batched=True)
eval_dataset = dataset['test'].map(tokenize_function, batched=True)

# Enlever les colonnes inutiles pour l'entraînement (garder uniquement input_ids, attention_mask, labels)
train_dataset = train_dataset.remove_columns(["text"])
eval_dataset = eval_dataset.remove_columns(["text"])

# Définir le format des datasets pour PyTorch (ou TensorFlow selon ton cas)
train_dataset.set_format("torch")
eval_dataset.set_format("torch")

# Charger le modèle BERT pré-entraîné pour la classification de séquence (nombre de labels = 2)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Définir les arguments d'entraînement (ajuste si nécessaire)
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",  # évalue à la fin de chaque époque
    per_device_train_batch_size=8,  # Ajuste selon les ressources GPU/CPU
    per_device_eval_batch_size=8,
    num_train_epochs=3,  # Nombre d'époques d'entraînement
    weight_decay=0.01  # Ajoute une régularisation si nécessaire
)

# Créer l'objet Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Lancer l'entraînement
trainer.train()