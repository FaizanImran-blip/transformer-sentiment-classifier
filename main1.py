from transformers import AutoModel
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = AutoModel.from_pretrained("bert-base-uncased").to(device)
model.eval()


def get_embeddings(tokens):
    print("Generating embeddings...")

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    return outputs.last_hidden_state
