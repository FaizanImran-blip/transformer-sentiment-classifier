import kagglehub
import pandas as pd
import os
import torch
import re
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import main1



# 1. CLEANING 

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text



2. LOAD DATASET

print("=" * 50)
print("STEP 1: LOADING DATA")
print("=" * 50)

path = kagglehub.dataset_download("krishbaisoya/tweets-sentiment-analysis")

train_df = pd.read_csv(os.path.join(path, "train_data.csv"))
test_df = pd.read_csv(os.path.join(path, "test_data.csv"))

print("Train columns:", train_df.columns)
print(f"Train size: {len(train_df)} rows")
print(f"Test size: {len(test_df)} rows")

# -------------------------
# 3. FIND TEXT COLUMN
# -------------------------
tweet_col = "sentence"
print(f"\nUsing column: '{tweet_col}'")

# -------------------------
# 4. CLEAN DATA
# -------------------------
print("\n" + "=" * 50)
print("STEP 2: CLEANING DATA")
print("=" * 50)

print("Cleaning training data...")
train_df["clean_text"] = train_df[tweet_col].apply(clean_text)

print("Cleaning test data...")
test_df["clean_text"] = test_df[tweet_col].apply(clean_text)


print("\nSample after cleaning:")
print(train_df[[tweet_col, "clean_text"]].head())

# -------------------------
# 5. LOAD TOKENIZER
# -------------------------
print("\n" + "=" * 50)
print("STEP 3: LOADING TOKENIZER")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
save_path = "/Users/Faizanimran/Downloads/final_model/tokenizer"
os.makedirs(save_path, exist_ok=True)
tokenizer.save_pretrained(save_path)
print("✅ Tokenizer loaded and saved!")

# -------------------------
# 6. TOKENIZATION WITH CACHING
# -------------------------
print("\n" + "=" * 50)
print("STEP 4: TOKENIZATION (WITH CACHING)")
print("=" * 50)

BATCH_SIZE = 120
MAX_LENGTH = 128

train_token_path = "/Users/Faizanimran/Downloads/final_model/train_tokens.pt"
test_token_path = "/Users/Faizanimran/Downloads/final_model/test_tokens.pt"


def tokenize_in_batches(texts, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_LENGTH):
    """Tokenize texts in batches to prevent memory freeze"""
    all_input_ids = []
    all_attention_masks = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Tokenizing", unit="batch"):
        batch_texts = texts[i : i + batch_size]

        tokens = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        all_input_ids.append(tokens["input_ids"])
        all_attention_masks.append(tokens["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0)
    attention_masks = torch.cat(all_attention_masks, dim=0)

    return {"input_ids": input_ids, "attention_mask": attention_masks}


# CHECK IF TOKENS ALREADY EXIST
if os.path.exists(train_token_path) and os.path.exists(test_token_path):
    print("✅ Loading cached tokens...")
    train_tokens = torch.load(train_token_path)
    test_tokens = torch.load(test_token_path)
else:
    print("⚡ Tokenizing now...")

    train_tokens = tokenize_in_batches(
        list(train_df["clean_text"]),
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    test_tokens = tokenize_in_batches(
        list(test_df["clean_text"]),
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )
    torch.save(train_tokens, train_token_path)
    torch.save(test_tokens, test_token_path)

    test_tokens = tokenize_in_batches(
        list(test_df["clean_text"]),
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    print("💾 Saving tokens to disk...")
    os.makedirs("/Users/Faizanimran/Downloads/final_model", exist_ok=True)
    torch.save(train_tokens, train_token_path)
    torch.save(test_tokens, test_token_path)
    print("✅ Tokens saved successfully!")

#. VERIFICATION


print("\n" + "=" * 50)
print("STEP 5: VERIFICATION")
print("=" * 50)

print(f"\n✅ Training tokens shape: {train_tokens['input_ids'].shape}")
print(f"✅ Test tokens shape: {test_tokens['input_ids'].shape}")
print(f"\nToken keys: {train_tokens.keys()}")


unique_labels = train_df["sentiment"].unique()
print(f"\n✅ Unique labels found: {unique_labels}")
num_classes = len(unique_labels)
print(f"✅ Number of classes: {num_classes}")

labels = torch.tensor(train_df["sentiment"].values)

dataset = TensorDataset(
    train_tokens["input_ids"], train_tokens["attention_mask"], labels
)


# -------------------------
# 8. USER INPUT FUNCTION
# -------------------------
def tokenize_user_input(text):
    """Tokenize user input for predictions"""
    text = clean_text(text)
    tokens = tokenizer(
        text, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )
    return tokens


# Example
print("\n" + "=" * 50)
print("STEP 6: TESTING USER INPUT")
print("=" * 50)

user_text = "I love this product!"
user_tokens = tokenize_user_input(user_text)

print(f"User input: {user_text}")
print(f"Cleaned: {clean_text(user_text)}")
print(f"Token IDs shape: {user_tokens['input_ids'].shape}")
print(f"Decoded: {tokenizer.decode(user_tokens['input_ids'][0])}")

print("\n" + "=" * 50)
print("🎉 SUCCESS! DATA → CLEANING → TOKENIZATION COMPLETE!")
print("=" * 50)


# -------------------------
# 9. EMBEDDING + MODEL (PHASE 7 - END MEIN)
# -------------------------
print("\n" + "=" * 50)
print("STEP 7: EMBEDDING & MODEL")
print("=" * 50)


sample_tokens = {
    "input_ids": train_tokens["input_ids"][:32],
    "attention_mask": train_tokens["attention_mask"][:32],
}


class SentimentModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        cls = x[:, 0, :]  
        return self.fc(cls)


model = SentimentModel(num_classes=num_classes)  
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print("✅ Model ready for training!")


print("\n" + "=" * 50)
print("STEP 8: TRAINING MODEL")
print("=" * 50)
if __name__ == "__main__":

    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

    for epoch in range(2):
        total_loss = 0
        correct = 0
        total = 0

        for batch in loader:
            input_ids, attention_mask, y = batch

            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            batch_tokens = {"input_ids": input_ids, "attention_mask": attention_mask}

            batch_embeddings = main1.get_embeddings(batch_tokens)
            batch_embeddings = batch_embeddings.to(device)

            outputs = model(batch_embeddings)

            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()


            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

        accuracy = 100 * correct / total
        print(
            f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}, Accuracy: {accuracy:.2f}%"
        )

    print("\n✅ Training complete!")
