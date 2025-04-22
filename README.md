
# Bhagavad Gita Verse Explanation Chatbot 🤖📖

This project is a chatbot powered by a fine-tuned [T5 Transformer](https://huggingface.co/transformers/model_doc/t5.html) that provides English explanations of verses from the **Bhagavad Gita**. It uses custom-formatted scripture data and NLP techniques to generate accurate, context-aware verse descriptions.

---

## 📌 Features

- 🔍 Verse-level explanations from Bhagavad Gita using deep learning.
- 🤖 Fine-tuned `t5-small` model for text-to-text generation.
- 📄 Custom PyTorch `Dataset` class for handling JSON/CSV-based inputs.
- 🧠 Uses Hugging Face `Trainer` for easy model training and evaluation.
- 💾 Supports saving and reloading of the model for future chatbot deployment.

---

## 🛠️ Tech Stack

- Python 3.x
- PyTorch
- Hugging Face Transformers
- Pandas
- scikit-learn
- Git LFS (for handling large model files)

---

## 📁 Project Structure

```
├── EG1.py                    # Model training and tokenizer script
├── fine_tuned_t5/            # Saved model and tokenizer directory
│   └── model.safetensors     # Fine-tuned T5 model file (requires Git LFS)
├── bhagavad_gita.json        # Training dataset (scripture-based)
├── bhagavad-gita1.csv        # Optional additional dataset
├── README.md                 # You're here!
```

---

## 🚀 How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/Bhagwat-geeta.git
   cd Bhagwat-geeta
   ```

2. Install dependencies:
   ```bash
   pip install torch transformers pandas scikit-learn
   ```

3. Run the training script:
   ```bash
   python EG1.py
   ```

---

## 📦 Using Git LFS

This repo contains large model files tracked using [Git Large File Storage (LFS)](https://git-lfs.github.com).

Before cloning:
```bash
git lfs install
```

When cloning:
```bash
git clone https://github.com/your-username/Bhagwat-geeta.git
```

Or after cloning, run:
```bash
git lfs pull
```

---

## 📤 Model Hosting (Optional)

If you’d rather not use Git LFS, you can upload `model.safetensors` to:
- Google Drive
- Hugging Face Hub
- Dropbox

...and modify your script to download it on the fly.

---

## 📚 Data Notes

- JSON input format:  
  ```json
  {
    "input": "Explain Chapter 2, Verse 47",
    "output": "You have a right to perform your prescribed duties..."
  }
  ```

- The CSV data (optional) is converted to the same structure internally.

---

## 🤝 Contributions

Feel free to fork, contribute, or suggest improvements via issues or pull requests.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

## 🙏 Credits

- Inspired by the Bhagavad Gita and classical Indian philosophy.
- Built with ❤️ using PyTorch and Hugging Face Transformers.
