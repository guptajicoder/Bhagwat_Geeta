import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Load the fine-tuned model
model_path = "./fine_tuned_t5"
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

# Load the JSON data
json_file_path = "combined_training_data.json"
with open(json_file_path, "r", encoding="utf-8") as f:
    training_data = json.load(f)

# Extract inputs and outputs
inputs = [entry["input"].lower() for entry in training_data]
outputs = [f"Chapter {entry['chapter']}, Verse {entry['verse']}: {entry['output']}" for entry in training_data]

# Precompute TF-IDF for the inputs
vectorizer = TfidfVectorizer()
input_vectors = vectorizer.fit_transform(inputs)

# Function to preprocess and vectorize the user input
def preprocess_and_vectorize(user_input):
    user_input = user_input.lower()
    user_vector = vectorizer.transform([user_input])
    return user_vector

# Chatbot loop
print("Chatbot: Namaste! How can I assist you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Chatbot: May peace be with you. Goodbye!")
        break

    # Vectorize the user input
    user_vector = preprocess_and_vectorize(user_input)

    # Find the most similar input in the dataset
    similarities = cosine_similarity(user_vector, input_vectors).flatten()
    best_match_idx = similarities.argmax()

    # If similarity is below a threshold, generate a generic response
    if similarities[best_match_idx] < 0.3:  # Adjust threshold as needed
        print("Chatbot: I am not sure how to help with that, but here's some guidance from the Bhagavad Gita:")
        inputs = tokenizer.encode(user_input, return_tensors="pt", truncation=True, max_length=512)
        outputs = model.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        response = outputs[best_match_idx]

    print(f"Chatbot: {response}")
