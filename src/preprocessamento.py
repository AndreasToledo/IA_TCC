import re
from torch.nn.utils.rnn import pad_sequence
import torch
 
MAX_PALAVRAS = 400
 
def limpar_texto(texto):
    texto = re.sub(r'[^a-zA-ZÀ-ÿ\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    texto = texto.strip().lower()
    return texto
 
def limitar_palavras(texto, max_palavras=MAX_PALAVRAS):
    palavras = texto.split()
    return ' '.join(palavras[:max_palavras])
 
def preprocessar_textos(lista_textos):
    textos_limpos = []
    for texto in lista_textos:
        texto = limpar_texto(texto)
        texto = limitar_palavras(texto)
        textos_limpos.append(texto)
    return textos_limpos
 
def construir_vocab(textos, min_freq=1):
    from collections import Counter
 
    counter = Counter()
    for texto in textos:
        palavras = texto.split()
        counter.update(palavras)
 
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for palavra, freq in counter.items():
        if freq >= min_freq:
            vocab[palavra] = len(vocab)
 
    return vocab
 
def textos_para_ids(textos, vocab):
    textos_ids = []
    for texto in textos:
        ids = [vocab.get(p, vocab["<UNK>"]) for p in texto.split()]
        tensor = torch.tensor(ids, dtype=torch.long)
        textos_ids.append(tensor)
    return textos_ids
 
def padronizar_lote(textos_ids, max_len=400):
    return pad_sequence(textos_ids, batch_first=True, padding_value=0)[:, :max_len]
 
import re
 
def preprocessar_texto(texto):
    """
    Limpa, normaliza e tokeniza o texto.
    """
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúâêîôûàèìòùç\s]', '', texto)  # remove pontuação
    texto = re.sub(r'\s+', ' ', texto).strip()
    tokens = texto.split()
    return tokens