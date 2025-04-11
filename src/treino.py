import pandas as pd
from torch.utils.data import Dataset

class MeuDatasetTexto(Dataset):
    def __init__(self, caminho_redacoes, caminho_notas, transform=None):
        self.redacoes = pd.read_csv(caminho_redacoes)
        self.notas = pd.read_csv(caminho_notas)
        self.transform = transform

        # Junta os dados com base no ID
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
