import os
from torch.utils.data import Dataset, DataLoader

class MeuDatasetTexto(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Caminho para a pasta com os textos (ex: dados/treino).
            transform (callable, optional): Transformação opcional no texto.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.textos = [os.path.join(root_dir, nome) for nome in os.listdir(root_dir)
                       if nome.lower().endswith(('.txt', '.csv'))]  # Filtra apenas arquivos de texto
    
    def __len__(self):
        """Retorna o número total de arquivos de texto no diretório."""
        return len(self.textos)

    def __getitem__(self, idx):
        """
        Carrega um arquivo de texto e o rótulo correspondente.
        
        Args:
            idx (int): Índice do arquivo a ser carregado.
        
        Returns:
            dict: Contém 'texto' e 'rótulo'.
        """
        caminho_arquivo = self.textos[idx]
        
        try:
            with open(caminho_arquivo, 'r', encoding='utf-8') as f:
                texto = f.read().strip()
        except Exception as e:
            print(f"Erro ao carregar {caminho_arquivo}: {e}")
            return None  # Pode ser tratado melhor dependendo do caso
        
        rótulo = self._extrair_rotulo(caminho_arquivo)
        
        if self.transform:
            texto = self.transform(texto)
        
        return {'texto': texto, 'rótulo': rótulo}
    
    def _extrair_rotulo(self, caminho_arquivo):
        """
        Extrai o rótulo do nome do arquivo ou da estrutura do diretório.
        Exemplo: Se o caminho for 'dados/treino/positivo/comentario1.txt', retorna 'positivo'.
        """