import os
import torch
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.modelo import criar_modelo
from src.preprocessamento import preprocessar_textos, textos_para_ids, padronizar_lote
 
# Configurações
MAX_PALAVRAS = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# Caminhos dos arquivos salvos
CAMINHO_MODELO = "resultados/checkpoints/modelo_treinado.pth"
CAMINHO_VOCAB = "resultados/checkpoints/vocab.pkl"
 
def processar_texto(texto):
    texto = texto.lower()
    return ' '.join(texto.split()[:MAX_PALAVRAS])
 
def carregar_dados_teste(caminho):
    redacoes = pd.read_csv(os.path.join(caminho, "redacoes.csv"))
    notas = pd.read_csv(os.path.join(caminho, "notas.csv"))
    textos = redacoes["texto"].tolist()
    rotulos = notas["nota"].tolist()
    return textos, rotulos
 
def avaliar_modelo(caminho_dados_teste="dados/teste"):
    # 1. Carregar o vocabulário
    with open(CAMINHO_VOCAB, "rb") as f:
        vocab = pickle.load(f)
 
    # 2. Carregar o modelo
    modelo = criar_modelo(vocab_size=len(vocab), output_dim=2)
    modelo.load_state_dict(torch.load(CAMINHO_MODELO, map_location=DEVICE))
    modelo.to(DEVICE)
    modelo.eval()
 
    # 3. Carregar os dados de teste
    textos, rotulos_reais = carregar_dados_teste(caminho_dados_teste)
    textos_proc = [processar_texto(t) for t in textos]
    textos_proc = preprocessar_textos(textos_proc)
    entradas = textos_para_ids(textos_proc, vocab)
    entradas_tensor = padronizar_lote(entradas, max_len=MAX_PALAVRAS).to(DEVICE)
 
    # 4. Fazer as predições
    with torch.no_grad():
        saida = modelo(entradas_tensor)
        preds = torch.argmax(saida, dim=1).cpu().numpy()
 
    # 5. Calcular métricas
    y_true = [int(r) for r in rotulos_reais]
    y_pred = preds
 
    print("📊 Relatório de Classificação:\n")
    print(classification_report(y_true, y_pred, digits=4))
    print(f"Acurácia: {accuracy_score(y_true, y_pred)*100:.2f}%")
    print(f"F1-score: {f1_score(y_true, y_pred, average='weighted'):.4f}")
 
if __name__ == "__main__":
    avaliar_modelo()