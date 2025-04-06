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
    É tipo assim, a gente vive num mundo que nem sempre é igual pra todo mundo né, e isso aí é complicado demais das conta. O negocio é que se todo mundo fosse feliz ia ser muito melhor do que não ser, e por isso que eu acho que a educação, ou seja lá o que for, tem que ser melhorada sim, porque senão não vai dar certo.

    Outra coisa é que às vezes o povo não tem o que fazer e fica inventando lei pra atrapalhar a vida dos outros, tipo não deixar ter barulho depois das 10, mas e se a pessoa trabalha de noite? Já pensou isso? Ninguém pensou. Por isso que o governo devia parar de gastar dinheiro com coisa atoa e começar a focar no que importa de verdade como, por exemplo, a comida que tá muito cara.

    E assim, com tudo isso, podemos dizer que a sociedade precisa pensar mais no que tá acontecendo, porque se não pensar, aí já viu né.
    """
    nota = avaliar_redacao(redacao_exemplo)
    print(f"Nota estimada: {nota}/1000")
