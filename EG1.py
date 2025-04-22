import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
import json
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure GPU usage if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the JSON file
json_file_path = "D:/Ram2/bhagavad_gita.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    json_data = json.load(f)

# # Load the CSV file
# csv_file_path = "D:/Ram2/bhagavad-gita1.csv"
# csv_df = pd.read_csv(csv_file_path)

# # Transform CSV data to match 'input' and 'output' structure
# csv_data = []
# for _, row in csv_df.iterrows():
#     chapter = row.get("Chapter", "Unknown Chapter")
#     verse = row.get("Verse", "Unknown Verse")
#     english_translation = row.get("English Translation", "No translation available.")
    
#     input_text = f"Explain Chapter {chapter}, Verse {verse}"
#     output_text = english_translation.strip()
    
#     csv_data.append({
#         "input": input_text,
#         "output": output_text,
#         "chapter": chapter,
#         "verse": verse,
#         "factor": "Verse Explanation"
#     })

# # Combine JSON and transformed CSV data
data = json_data #+ csv_data


# Validate dataset structure
for idx, item in enumerate(data):
    if "input" not in item or "output" not in item:
        print(f"Missing keys in item at index {idx}: {item}")
        item["input"] = item.get("input", "Default input text")
        item["output"] = item.get("output", "Default output text")

# Initialize tokenizer and model
model_name = "t5-small"  # You can use 't5-base' or 't5-large' for better performance
tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)  # Move model to GPU

# Define custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item.get("input", "Default input text")
        output_text = item.get("output", "Default output text")
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        outputs = self.tokenizer(
            output_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": outputs["input_ids"].squeeze(0),
        }

# Split data into training and evaluation datasets
train_data, eval_data = train_test_split(data, test_size=0.1, random_state=42)  # 10% for evaluation

# Create datasets
train_dataset = CustomDataset(train_data, tokenizer)
eval_dataset = CustomDataset(eval_data, tokenizer)

# Custom data collator to ensure tensors are not pinned
def custom_data_collator(features):
    batch = {
        "input_ids": torch.stack([f["input_ids"] for f in features]),
        "attention_mask": torch.stack([f["attention_mask"] for f in features]),
        "labels": torch.stack([f["labels"] for f in features]),
    }
    return batch

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results_t5",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=0,
    logging_dir="./logs_t5",
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_strategy="no",
    report_to="none",
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=custom_data_collator,
)

# Train the model
print("Training started...")
trainer.train()
metrics = trainer.evaluate()

# Save the fine-tuned model
model.save_pretrained("./fine_tuned_t5")
tokenizer.save_pretrained("./fine_tuned_t5")

print("Fine-tuned model saved successfully!")
