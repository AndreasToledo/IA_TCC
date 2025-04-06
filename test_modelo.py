import torch
from src.modelo import criar_modelo

# Criar modelo
modelo = criar_modelo(vocab_size=5000).eval()

# Simular uma entrada fictícia (batch de 2 redações com 10 tokens cada)
entrada_teste = torch.randint(0, 5000, (2, 10))

# Obter a saída
saida = modelo(entrada_teste)

print("Saída do modelo:", saida)
