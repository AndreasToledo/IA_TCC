import torch
import pickle
from src.modelo import criar_modelo
from src.preprocessamento import preprocessar_texto

modelo_path = "resultados/checkpoints/modelo_treinado.pth"
vocab_path = "resultados/checkpoints/vocab.pkl"
 
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)
 
modelo = criar_modelo(vocab_size=len(vocab))
modelo.load_state_dict(torch.load(modelo_path))
modelo.eval()
 
def texto_para_ids(texto, vocab, max_len=100):
    tokens = preprocessar_texto(texto)
    ids = [vocab.get(token, vocab.get("<UNK>", 0)) for token in tokens]
    ids = ids[:max_len] + [0] * max(0, max_len - len(ids))  # padding
    return torch.tensor([ids], dtype=torch.long)  # garante tipo correto
 
def avaliar_redacao(texto):
    entrada = texto_para_ids(texto, vocab)
    with torch.no_grad():
        saida = modelo(entrada)
        nota_bruta = saida.item() * 1000

        num_palavras = len(texto.split())
        if num_palavras < 80:
            fator_penalidade = num_palavras / 300
            nota_ajustada = nota_bruta * fator_penalidade
        else:
            nota_ajustada = nota_bruta

        return round(nota_ajustada, 2)

 
if __name__ == "__main__":
    redacao_exemplo = """
A leitura é uma das atividades mais enriquecedoras que um indivíduo pode realizar. Ela desempenha um papel fundamental no desenvolvimento pessoal, cultural e social de cada pessoa, oferecendo oportunidades de aprendizado, reflexão e entretenimento. Apesar disso, seu valor ainda é subestimado por muitos, especialmente em um mundo cada vez mais dominado por tecnologias digitais.

Em primeiro lugar, a leitura amplia nosso conhecimento sobre o mundo. Por meio de livros, revistas e artigos, podemos explorar temas variados, entender diferentes culturas e aprender sobre ciências, história, filosofia e muito mais. Além disso, ela desenvolve nossa capacidade crítica, já que somos incentivados a analisar e interpretar informações, tornando-nos cidadãos mais conscientes e informados.

Além de nos tornar mais informados, a leitura também exerce benefícios para nossa saúde mental. Estudos indicam que dedicar tempo à leitura reduz o estresse e promove o relaxamento. Ler ficção, por exemplo, permite que mergulhemos em histórias e vivamos experiências por meio dos personagens, o que pode ser um alívio para os desafios do dia a dia. Já textos não ficcionais podem nos oferecer soluções práticas e inspiração para superar problemas.

Por fim, é essencial destacar que a leitura é uma ferramenta poderosa de inclusão e transformação social. Através do acesso a livros e a outros recursos escritos, indivíduos têm a oportunidade de mudar suas realidades, adquirir habilidades para o mercado de trabalho e até melhorar sua autoestima. Isso reforça a importância de políticas públicas voltadas à democratização do acesso à literatura e ao incentivo à leitura em comunidades menos favorecidas.

Em conclusão, a leitura é um hábito que deve ser valorizado e incentivado. Ela não apenas nos conecta a mundos diferentes, mas também nos enriquece como seres humanos. Investir tempo na leitura é, sem dúvida, investir em nosso próprio desenvolvimento.
    """
    nota = avaliar_redacao(redacao_exemplo)
    print(f"Nota estimada: {nota}/1000")