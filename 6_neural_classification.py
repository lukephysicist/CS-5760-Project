import json
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# load in the student message data from part 1
records = [json.loads(line) for line in open("data/4_student_message.jsonl")]
df = pd.DataFrame(records)

#this gives it a numerical label in addtion to the string label
label_enc = LabelEncoder()
df["label_id"] = label_enc.fit_transform(df["label"])


#### ENCODING, these create numerical values for each token
def tokenize(text): return text.lower().split()

counter = Counter(tok for text in df["text"] for tok in tokenize(text))
vocab = {"<PAD>": 0, "<UNK>": 1, **{w: i+2 for i, (w, _) in enumerate(counter.most_common(10000))}}

def encode(text): return [vocab.get(t, 1) for t in tokenize(text)]


# A data class that extends the pytorch dataset api
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.x = [torch.tensor(encode(t)) for t in texts]
        self.y = torch.tensor(labels.values, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.x[i], self.y[i]

def collate_fn(batch):
    xs, ys = zip(*batch)
    return pad_sequence(xs, batch_first=True, padding_value=0), torch.stack(ys)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_loader = DataLoader(TextDataset(train_df["text"], train_df["label_id"]), batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader  = DataLoader(TextDataset(test_df["text"],  test_df["label_id"]),  batch_size=32, collate_fn=collate_fn)


# Our model class
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc        = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        return self.fc(hidden.squeeze(0))

num_classes = df["label_id"].nunique()
model = RNNClassifier(vocab_size=len(vocab), embed_dim=64, hidden_dim=128, num_classes=num_classes)

# ── 5. Training ───────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss, correct = 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item()
            correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(loader), correct / len(loader.dataset)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
for epoch in range(10):
    tl, ta = run_epoch(train_loader, train=True)
    vl, va = run_epoch(test_loader,  train=False)
    for k, v in zip(history, [tl, vl, ta, va]): history[k].append(v)
    print(f"Epoch {epoch+1:02d} | loss {tl:.3f}/{vl:.3f} | acc {ta:.3f}/{va:.3f}")

# ── 6. Training Curves ────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(history["train_loss"], label="Train"); ax1.plot(history["val_loss"], label="Val")
ax1.set(title="Loss", xlabel="Epoch"); ax1.legend()
ax2.plot(history["train_acc"], label="Train"); ax2.plot(history["val_acc"], label="Val")
ax2.set(title="Accuracy", xlabel="Epoch"); ax2.legend()
plt.tight_layout(); plt.savefig("training_curves.png"); plt.show()

print(f"\nFinal Test Accuracy: {history['val_acc'][-1]:.4f}")