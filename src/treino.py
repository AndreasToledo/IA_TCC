import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
from src.dataset import MeuDatasetTexto
from src.modelo import criar_modelo
from src.preprocessamento import (
    preprocessar_textos,
    construir_vocab,
    textos_para_ids,
    padronizar_lote
)

# Configurações
EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_PALAVRAS = 400
OUTPUT_DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Caminhos
CAMINHO_TREINO = "dados/treino"
CAMINHO_MODELO = "resultados/checkpoints/modelo_treinado.pth"
CAMINHO_VOCAB = "resultados/checkpoints/vocab.pkl"

def transformar_texto_em_tensor(textos, vocab):
    ids = textos_para_ids(textos, vocab)
    return padronizar_lote(ids, max_len=MAX_PALAVRAS)

def treinar_modelo():
    print("Carregando dados...")
    
    def processar_texto(texto):
        texto = texto.lower()
        return ' '.join(texto.split()[:MAX_PALAVRAS])  # Limita a 400 palavras

    # Carregar dataset
    dataset = MeuDatasetTexto(root_dir=CAMINHO_TREINO, transform=processar_texto)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Coleta os textos do dataset para construir o vocabulário
    todos_os_textos = [processar_texto(entrada['texto']) for entrada in dataset]
    textos_pre = preprocessar_textos(todos_os_textos)
    vocab = construir_vocab(textos_pre)

    print(f"Vocabulário construído com {len(vocab)} palavras.")

    # Criar modelo
    modelo = criar_modelo(vocab_size=len(vocab), output_dim=OUTPUT_DIM).to(DEVICE)
    criterio = nn.CrossEntropyLoss()
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    print("Iniciando o treinamento...")
    for epoch in range(EPOCHS):
        modelo.train()
        total_loss = 0

        for batch in dataloader:
            textos = [processar_texto(t) for t in batch['texto']]
            rotulos = [0 if r == 'negativo' else 1 for r in batch['rótulo']]

            entradas = transformar_texto_em_tensor(textos, vocab).to(DEVICE)
            rotulos = torch.tensor(rotulos, dtype=torch.long).to(DEVICE)

            otimizador.zero_grad()
            saida = modelo(entradas)
            loss = criterio(saida, rotulos)
            loss.backward()
            otimizador.step()

            total_loss += loss.item()

        print(f"Época {epoch+1}/{EPOCHS} - Loss: {total_loss / len(dataloader):.4f}")

    print("Treinamento concluído!")

    # Criar pasta se não existir
    os.makedirs(os.path.dirname(CAMINHO_MODELO), exist_ok=True)

    # Salvar modelo
    torch.save(modelo.state_dict(), CAMINHO_MODELO)
    print(f"Modelo salvo em: {CAMINHO_MODELO}")

    # Salvar vocabulário
    with open(CAMINHO_VOCAB, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulário salvo em: {CAMINHO_VOCAB}")

# Executa o treinamento
if __name__ == "__main__":
    treinar_modelo()