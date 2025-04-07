# inferencia.py (como já tá estruturado)

import torch
import pickle
from src.modelo import criar_modelo
from src.preprocessamento import preprocessar_texto  # vamos criar agora

# Caminhos
modelo_path = "resultados/checkpoints/modelo_treinado.pth"
vocab_path = "resultados/checkpoints/vocab.pkl"

with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)

modelo = criar_modelo(vocab_size=len(vocab))
modelo.load_state_dict(torch.load(modelo_path))
modelo.eval()


# Converte texto em tensor
def texto_para_ids(texto, vocab, max_len=100):
    tokens = preprocessar_texto(texto)
    ids = [vocab.get(token, vocab.get("<UNK>", 0)) for token in tokens]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))  # padding
    return torch.tensor([ids])  # forma (1, max_len)

# Função principal
def avaliar_redacao(texto):
    entrada = texto_para_ids(texto, vocab)
    with torch.no_grad():
        saida = modelo(entrada)
        nota_normalizada = saida.item()
        nota_1000 = round(nota_normalizada * 1000, 2)
        return nota_1000

# Teste de exemplo
if __name__ == "__main__":
    redacao_exemplo = """
    Hoje em dia tem muita tecnologia e isso muda as coisas. As pessoas tão usando celular todo dia e isso tá mudando como elas vivem, porque tudo ficou mais fácil e rápido. Antes era diferente, agora todo mundo fica no telefone, até as crianças.

A tecnologia é boa, mas também tem problema. Tem gente que fica viciada e não faz mais nada da vida. Tem que saber usar. A internet ajuda a estudar e a conversar com as pessoas de longe, mas também pode atrapalhar.

O governo precisa fazer alguma coisa, tipo ensinar a usar a internet direito. Porque tem muita gente que usa errado. Tem que ter regras.

Então a tecnologia é boa, mas ruim também. Tem que usar com cuidado.
    """
    nota = avaliar_redacao(redacao_exemplo)
    print(f"Nota estimada: {nota}/1000")
