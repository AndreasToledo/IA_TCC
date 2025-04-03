import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.dataset import MeuDatasetTexto
from src.modelo import criar_modelo

# Configuração
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001
VOCAB_SIZE = 5000  # Deve ser ajustado de acordo com o vocabulário real
OUTPUT_DIM = 2  # Número de classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transformar_texto_em_tensor(textos):
    """
    Converte uma lista de textos em tensores numéricos.
    Aqui você pode usar um tokenizador como o da Hugging Face, spaCy ou TF-IDF.
    """
    vocab = {'bom': 1, 'ruim': 2, 'ótimo': 3, 'péssimo': 4}  # Vocabulário fictício
    max_len = 10  # Comprimento fixo para padding/truncamento
    tensors = []

    for texto in textos:
        tokens = [vocab.get(palavra, 0) for palavra in texto.split()]  # Mapeia palavras para IDs
        tokens = tokens[:max_len] + [0] * (max_len - len(tokens))  # Padding
        tensors.append(tokens)

    return torch.tensor(tensors, dtype=torch.long)

# Função para treinar o modelo
def treinar_modelo():
    print("Carregando dados...")
    caminho_treino = "dados/treino"

    def processar_texto(texto):
        return texto.lower()

    dataset = MeuDatasetTexto(root_dir=caminho_treino, transform=processar_texto)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    modelo = criar_modelo(vocab_size=VOCAB_SIZE, output_dim=OUTPUT_DIM).to(DEVICE)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    print("Iniciando o treinamento...")
    for epoch in range(EPOCHS):
        total_loss = 0
        modelo.train()
        
        for batch in dataloader:
            textos = batch['texto']
            rotulos = batch['rótulo']

            entradas = transformar_texto_em_tensor(textos).to(DEVICE)
            rotulos = torch.tensor([0 if r == 'negativo' else 1 for r in rotulos], dtype=torch.long).to(DEVICE)

            otimizador.zero_grad()
            saida = modelo(entradas)
            loss = criterio(saida, rotulos)
            loss.backward()
            otimizador.step()

            total_loss += loss.item()
        
        print(f"Época {epoch+1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    print("Treinamento concluído! Salvando modelo...")
    torch.save(modelo.state_dict(), "resultados/checkpoints/modelo_treinado.pth")
    print("Modelo salvo com sucesso!")

# Executa o treinamento
if __name__ == "__main__":
    treinar_modelo()
