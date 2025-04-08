import torch
import torch.nn as nn

class ModeloRegressaoTexto(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super(ModeloRegressaoTexto, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Saída única para nota (regressão)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])  # Sem ativação — saída pode ser entre 0 e 1000
        return out

def criar_modelo(vocab_size=5000, embed_dim=100, hidden_dim=128):
    return ModeloRegressaoTexto(vocab_size, embed_dim, hidden_dim)
