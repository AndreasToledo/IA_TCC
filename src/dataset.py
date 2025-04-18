import pandas as pd
from torch.utils.data import Dataset

caminho_redacoes = "dados/treino/redacoes.csv"
caminho_notas = "dados/treino/notas.csv"

class MeuDatasetTexto(Dataset):
    def __init__(self, caminho_redacoes, caminho_notas, transform=None):
        """
        Args:
            caminho_redacoes (str): Caminho para o CSV com id,texto
            caminho_notas (str): Caminho para o CSV com id,nota (valores de 0.1 a 1.0)
            transform (callable, optional): Transformação opcional no texto.
        """
        self.transform = transform

        df_textos = pd.read_csv(caminho_redacoes)
        df_notas = pd.read_csv(caminho_notas)
        self.df = pd.merge(df_textos, df_notas, on="id")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        texto = self.df.iloc[idx]["texto"]
        nota = self.df.iloc[idx]["nota"]

        if self.transform:
            texto = self.transform(texto)

        return {"texto": texto, "rótulo": nota}