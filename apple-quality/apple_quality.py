import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from scipy.stats import zscore
from sklearn.model_selection import train_test_split 


if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

df = pd.read_csv('apple_quality.csv')
del df['A_id']
df = df.iloc[0:3999, :]
columns = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity", "Quality"]

df['Acidity'] = df['Acidity'].astype('float')
df["Quality"] = df['Quality'].replace({"good":1,"bad":0})
X = df.drop('Quality', axis=1)
y = df['Quality']

#Standardization
for col in X:
    X[col] = zscore(X[col])

print("Data Info: \n")
print(X.describe(), '\n\nFirst few columns as an example:\n\n')
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, shuffle=True)

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.from_numpy(features.values).float().to(device)
        self.labels = torch.from_numpy(labels.values).long().to(device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

training_data = CustomDataset(X_train, y_train)
test_data = CustomDataset(X_test, y_test)

batch_size = 64
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(7, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
model = NeuralNetwork()

def train_loop(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            if (t +1) % 10 == 0:
                print(f"loss: {loss:>7f}   [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn, t):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    if (t +1) % 10 == 0:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

lr = 1e-2
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = lr)

epochs = 200
for t in range(epochs):
    if (t +1) % 10 == 0:
        print(f"Epoch {t+1}\n-----------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer, t)
    test_loop(test_dataloader, model, loss_fn, t)
print("Done!")