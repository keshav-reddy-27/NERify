# Install required libraries (if not already installed)
# !pip install datasets transformers torch

import os
import torch # type: ignore
from datasets import load_dataset # type: ignore
from transformers import (
    DistilBertTokenizerFast, DistilBertForTokenClassification, 
    Trainer, TrainingArguments, DataCollatorForTokenClassification
)

# Disable WandB logging
os.environ["WANDB_DISABLED"] = "true"

# Load CoNLL-2003 dataset
dataset = load_dataset("conll2003", trust_remote_code=True)


# Use 50% of training data
train_data = dataset["train"].select(range(int(len(dataset["train"]) * 0.5)))
valid_data = dataset["validation"]

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

# Function to tokenize and align labels
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding="max_length",
        max_length=128,
        is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        prev_word_id = None
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)  # Ignore padding
            elif word_id != prev_word_id:
                label_ids.append(label[word_id])
            else:
                label_ids.append(-100)  # Ignore subword tokens
            prev_word_id = word_id
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Apply tokenization
train_dataset = train_data.map(tokenize_and_align_labels, batched=True)
valid_dataset = valid_data.map(tokenize_and_align_labels, batched=True)

# Get the number of entity labels
num_labels = len(dataset["train"].features["ner_tags"].feature.names)

# Load model
model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=2,  # Reduced batch size for lower memory usage
    per_device_eval_batch_size=2,
    num_train_epochs=3,  # Increased training epochs
    weight_decay=0.01,
    report_to="none"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# Train the model
#trainer.train()

# Save the model
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")

# Load entity labels
label_list = dataset["train"].features["ner_tags"].feature.names



# Example test
user_input = "Barack Obama was born in Honolulu, Hawaii. He studied at Harvard University and later joined politics. Microsoft and Google are top companies."
predicted_entities = predict_entities(user_input)

print("\nRecognized Entities:")
for entity, entity_type, confidence in predicted_entities:
    print(f"Entity: {entity}")
    print(f"Type: {entity_type}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 30)
# Save the trained model and tokenizer
model.save_pretrained("./ner_model")
tokenizer.save_pretrained("./ner_model")
