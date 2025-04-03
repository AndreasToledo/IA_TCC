import argparse
import torch
from src.treino import treinar_modelo
from src.avaliacao import avaliar_modelo
from src.modelo import criar_modelo
from src.dataset import MeuDatasetTexto  # Se precisar para testes

# Configurações
VOCAB_SIZE = 5000  # Ajuste conforme seu vocabulário real
OUTPUT_DIM = 2  # Número de classes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para testar uma frase no modelo treinado
def testar_texto(frase):
    modelo = criar_modelo(vocab_size=VOCAB_SIZE, output_dim=OUTPUT_DIM).to(DEVICE)
    modelo.load_state_dict(torch.load("resultados/checkpoints/modelo_treinado.pth", map_location=DEVICE))
    modelo.eval()

    # Transformar texto em tensor (isso pode mudar dependendo da sua tokenização)
    tokens = [1 if palavra == "bom" else 2 if palavra == "ruim" else 0 for palavra in frase.lower().split()]
    max_len = 10
    tokens = tokens[:max_len] + [0] * (max_len - len(tokens))
    entrada = torch.tensor([tokens], dtype=torch.long).to(DEVICE)

    # Predição
    with torch.no_grad():
        saida = modelo(entrada)
        previsao = torch.argmax(saida, dim=1).item()

    classe = "positivo" if previsao == 1 else "negativo"
    print(f"Texto: '{frase}' → Predição: {classe}")

# Argumentos do CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executar modelo de análise de texto")
    parser.add_argument("--train", action="store_true", help="Treinar o modelo")
    parser.add_argument("--eval", action="store_true", help="Avaliar o modelo")
    parser.add_argument("--test", type=str, help="Testar uma frase específica")

    args = parser.parse_args()

    if args.train:
        treinar_modelo()
    elif args.eval:
        avaliar_modelo()
    elif args.test:
        testar_texto(args.test)
    else:
        print("Use --train para treinar, --eval para avaliar ou --test 'frase' para testar.")
