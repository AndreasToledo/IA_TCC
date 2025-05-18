import sys
import torch
import pickle
import io
from src.modelo import criar_modelo
from src.preprocessamento import preprocessar_texto

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

modelo_path = r"C:\Users\User\Documents\TCC\IA_TCC\resultados\checkpoints\modelo_treinado.pth"
vocab_path = r"C:\Users\User\Documents\TCC\IA_TCC\resultados\checkpoints\vocab.pkl"
                
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

modelo = criar_modelo(vocab_size=len(vocab))
modelo.load_state_dict(torch.load(modelo_path))
modelo.eval()

def texto_para_ids(texto, vocab, max_len=100):
    tokens = preprocessar_texto(texto)
    ids = [vocab.get(token, vocab.get("<UNK>", 0)) for token in tokens]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))
    return torch.tensor([ids], dtype=torch.long)

def avaliar_redacao(texto):
    entrada = texto_para_ids(texto, vocab)
    with torch.no_grad():
        saida = modelo(entrada)
        nota_bruta = saida.item() * 1000
        nota_bruta = min(nota_bruta, 1000)

        num_palavras = len(texto.split())
        if num_palavras < 80:
            fator_penalidade = num_palavras / 300
            nota_ajustada = nota_bruta * fator_penalidade
        else:
            nota_ajustada = nota_bruta

        return round(nota_ajustada, 2)

if __name__ == "__main__":
    texto_entrada = sys.stdin.read().strip()
    nota = avaliar_redacao(texto_entrada)
    print(nota)