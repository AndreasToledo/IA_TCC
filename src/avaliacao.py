import torch
from src.dataset import MeuDatasetTexto  # Agora importamos a versão de texto do dataset
from src.modelo import criar_modelo
from torch.utils.data import DataLoader

def avaliar_modelo():
    # Configuração inicial
    caminho_teste = "dados/teste"

    def processar_texto(texto):
        """Função de pré-processamento do texto (exemplo: converter para minúsculas)."""
        return texto.lower()

    dataset = MeuDatasetTexto(root_dir=caminho_teste, transform=processar_texto)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    modelo = criar_modelo()
    modelo.load_state_dict(torch.load("resultados/checkpoints/modelo_treinado.pth"))
    modelo.eval()

    total = 0
    corretos = 0

    with torch.no_grad():
        for batch in dataloader:
            textos = batch['texto']
            rotulos = batch['rótulo']

            # Aqui você precisa transformar os