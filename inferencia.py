import torch
import pickle
from src.modelo import criar_modelo
from src.preprocessamento import preprocessar_texto
 
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
    return torch.tensor([ids], dtype=torch.long)  # garante tipo correto
 
def avaliar_redacao(texto):
    entrada = texto_para_ids(texto, vocab)
    with torch.no_grad():
        saida = modelo(entrada)
        nota_1000 = saida.item() * 1000
        return round(nota_1000, 2)
 
# Teste de exemplo
if __name__ == "__main__":
    redacao_exemplo = """
    A construção de uma sociedade verdadeiramente justa pressupõe, além de políticas públicas efetivas, o fortalecimento de valores éticos que orientem as relações humanas. Nesse contexto, a empatia — a capacidade de se colocar no lugar do outro — configura-se como elemento central para a promoção do respeito mútuo, da solidariedade e da inclusão. No entanto, observa-se, no Brasil contemporâneo, um déficit significativo desse valor, o que contribui para a manutenção de desigualdades e da intolerância social. Tal realidade exige uma abordagem multidimensional, que envolva a educação, a comunicação e o exemplo vindo das esferas de poder.
 
Primeiramente, é preciso considerar que a ausência de empatia se enraíza, em grande medida, na formação histórica e cultural do país, marcado por profundas desigualdades sociais. A naturalização do sofrimento alheio, perceptível em discursos meritocráticos e em práticas de exclusão, revela a necessidade urgente de fomentar uma cultura de escuta e reconhecimento da alteridade. Nesse sentido, o filósofo Martin Buber já afirmava que a verdadeira humanidade só se manifesta quando o “Eu” reconhece o “Tu” em sua plenitude, sem reduções ou estigmas.
 
Ademais, os meios de comunicação e as redes sociais possuem papel ambíguo nesse processo. Enquanto podem ser ferramentas de empatia — ao dar visibilidade a vozes silenciadas — também fomentam a polarização e a desumanização do outro, especialmente quando dominadas por discursos de ódio e fake news. Dessa forma, é imperativo promover uma media literacy nas escolas, que capacite os jovens a compreender criticamente o conteúdo que consomem e compartilham.
 
Por fim, cabe destacar que o exemplo vindo das lideranças políticas e institucionais exerce enorme influência sobre o comportamento coletivo. Um governante que naturaliza a exclusão ou que ridiculariza minorias legitima o desrespeito como prática cotidiana. Por isso, é essencial que as autoridades adotem posturas éticas e responsáveis, alinhadas ao princípio da dignidade humana.
 
Portanto, para que a empatia deixe de ser apenas um ideal abstrato e se converta em prática social concreta, é necessário investir em uma educação humanizadora, em políticas de combate à desigualdade e em lideranças comprometidas com o bem comum. Só assim será possível edificar uma sociedade onde o “nós” prevaleça sobre o “eu”.
    """
    nota = avaliar_redacao(redacao_exemplo)
    print(f"Nota estimada: {nota}/1000")