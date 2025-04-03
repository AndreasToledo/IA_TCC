import torch
import torch.nn as nn

class ModeloTexto(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128, output_dim=2):
        """
        Modelo baseado em embeddings + LSTM para classificação de texto.
        
        Args:
            vocab_size (int): Tamanho do vocabulário.
            embed_dim (int): Dimensão dos embeddings.
            hidden_dim (int): Dimensão da camada oculta do LSTM.
            output_dim (int): Número de classes para predição.
        """
        super(ModeloTexto, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)  # Converte IDs para embeddings
        _, (hidden, _) = self.lstm(x)  # Pegamos a última camada oculta do LSTM
        out = self.fc(hidden[-1])  # Mapeia para as classes
        return self.softmax(out)

def criar_modelo(vocab_size=5000, embed_dim=100, hidden_dim=128, output_dim=2):
    """
    Define e retorna um modelo baseado em LSTM para análise de texto.
    
    Args:
        vocab_size (int): Tamanho do vocabulário.
        embed_dim (int): Dimensão dos embeddings.
        hidden_dim (int): Tamanho da camada oculta do LSTM.
        output_dim (int): Número de classes.

    Returns:
        Modelo de NLP baseado em LSTM.
    """
    return ModeloTexto(vocab_size, embed_dim, hidden_dim, output_dim)

# Testando
if __name__ == "__main__":
    modelo = criar_modelo()
    print(modelo)
