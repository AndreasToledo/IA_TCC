import os
from torch.utils.data import Dataset

class MeuDatasetTexto(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Caminho para a pasta com os textos (ex: dados/treino).
            transform (callable, optional): Transformação opcional no texto.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.arquivos = [nome for nome in os.listdir(root_dir)
                         if nome.lower().endswith(('.txt', '.csv'))]

    def __len__(self):
        return len(self.arquivos)

    def __getitem__(self, idx):
        arquivo = self.arquivos[idx]
        caminho = os.path.join(self.root_dir, arquivo)

        try:
            with open(caminho, 'r', encoding='utf-8') as f:
                texto = f.read().strip()
        except Exception as e:
            print(f"Erro ao ler {caminho}: {e}")
            texto = "[ERRO]"

        if self.transform:
            texto = self.transform(texto)

        if not texto:
            texto = "[VAZIO]"

        rotulo = self._extrair_rotulo(caminho)

        return {"texto": texto, "rótulo": rotulo}

    def _extrair_rotulo(self, caminho_arquivo):
        nome = os.path.basename(caminho_arquivo).lower()

        if "bom" in nome:
            return "positivo"
        elif "ruim" in nome:
            return "negativo"

        pasta_pai = os.path.basename(os.path.dirname(caminho_arquivo)).lower()
        if pasta_pai in ["positivo", "negativo"]:
            return pasta_pai

        return "desconhecido"
