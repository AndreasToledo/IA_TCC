import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim  
import pickle
import os
from modelo import ModeloRegressaoTexto
from torch.utils.data import DataLoader, Dataset

EPOCHS = 10
BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_PALAVRAS = 400
OUTPUT_DIM = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAMINHO_TREINO = "dados/treino"
CAMINHO_MODELO = "resultados/checkpoints/modelo_treinado.pth"
CAMINHO_VOCAB = "resultados/checkpoints/vocab.pkl"

def processar_texto(texto):
    texto = texto.lower()
    return ' '.join(texto.split()[:MAX_PALAVRAS])  

def preprocessar_textos(textos):
    return textos

def construir_vocab(textos):
    vocab = {}
    for texto in textos:
        for palavra in texto.split():
            if palavra not in vocab:
                vocab[palavra] = len(vocab)
    return vocab

def textos_para_ids(textos, vocab):
    return [[vocab.get(palavra, 0) for palavra in texto.split()] for texto in textos]

def padronizar_lote(lote, max_len):
    padded_tensor = torch.zeros(len(lote), max_len, dtype=torch.long)
    
    for i, seq in enumerate(lote):
        seq_len = min(len(seq), max_len)
        padded_tensor[i, :seq_len] = torch.tensor(seq[:seq_len], dtype=torch.long)
    
    return padded_tensor


class MeuDatasetTexto(Dataset):
    def __init__(self, caminho_redacoes, caminho_notas, transform=None):
        self.redacoes = pd.read_csv(caminho_redacoes)
        self.notas = pd.read_csv(caminho_notas)
        self.transform = transform
        self.dados = pd.merge(self.redacoes, self.notas, on="id")

    def __len__(self):
        return len(self.dados)

    def __getitem__(self, idx):
        linha = self.dados.iloc[idx]
        texto = linha["texto"]
        nota = linha["nota"]

        if self.transform:
            texto = self.transform(texto)

        return {"texto": texto, "rótulo": nota}

def transformar_texto_em_tensor(textos, vocab):
    ids = textos_para_ids(textos, vocab)
    return padronizar_lote(ids, max_len=MAX_PALAVRAS)


def treinar_modelo():
    print("Iniciando o treinamento...")

    dataset = MeuDatasetTexto(caminho_redacoes=os.path.join(CAMINHO_TREINO, "redacoes.csv"),
                               caminho_notas=os.path.join(CAMINHO_TREINO, "notas.csv"),
                               transform=processar_texto)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    todos_os_textos = [processar_texto(entrada['texto']) for entrada in dataset]
    textos_pre = preprocessar_textos(todos_os_textos)
    vocab = construir_vocab(textos_pre)

    print(f"Vocabulário construído com {len(vocab)} palavras.")

    modelo = ModeloRegressaoTexto(vocab_size=len(vocab)).to(DEVICE)

    criterio = nn.MSELoss()
    otimizador = optim.Adam(modelo.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        modelo.train()
        total_loss = 0

        for batch in dataloader:
            textos = [processar_texto(t) for t in batch['texto']]
            rotulos = [0 if r == 'negativo' else 1 for r in batch['rótulo']]

            entradas = transformar_texto_em_tensor(textos, vocab).to(DEVICE)
            rotulos = torch.tensor(rotulos, dtype=torch.float).unsqueeze(1).to(DEVICE)

            otimizador.zero_grad()
            saida = modelo(entradas)
            loss = criterio(saida, rotulos)
            loss.backward()
            otimizador.step()

            total_loss += loss.item()

        print(f"Época {epoch+1}/{EPOCHS} - Loss: {total_loss / len(dataloader):.4f}")

    print("Treinamento concluído!")

    os.makedirs(os.path.dirname(CAMINHO_MODELO), exist_ok=True)
    torch.save(modelo.state_dict(), CAMINHO_MODELO)
    print(f"Modelo salvo em: {CAMINHO_MODELO}")

    with open(CAMINHO_VOCAB, 'wb') as f:
        pickle.dump(vocab, f)
    print(f"Vocabulário salvo em: {CAMINHO_VOCAB}")

if __name__ == "__main__":
    treinar_modelo()
