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
        nota_bruta = saida.item() * 1000

        # Penaliza textos muito curtos com maior intensidade (agora linear)
        num_palavras = len(texto.split())
        if num_palavras < 80:
            fator_penalidade = num_palavras / 300
            nota_ajustada = nota_bruta * fator_penalidade
        else:
            nota_ajustada = nota_bruta

        return round(nota_ajustada, 2)

 
# Teste de exemplo
if __name__ == "__main__":
    redacao_exemplo = """
   A Constituição Federal de 1988 assegura, em seu artigo 5º, que todos são iguais perante a lei, sem distinção de qualquer natureza. No entanto, a realidade brasileira revela um abismo entre a teoria jurídica e a prática social, especialmente no que se refere à inclusão de pessoas com deficiência. Apesar dos avanços legislativos, como a Lei Brasileira de Inclusão (2015), a efetivação da acessibilidade e da integração plena ainda é um desafio que persiste.

Em primeiro lugar, é notável a precariedade na infraestrutura urbana e nos espaços públicos, que frequentemente não atendem às necessidades das pessoas com deficiência. Calçadas irregulares, ausência de rampas, transporte coletivo não adaptado e a falta de sinalização tátil são obstáculos diários que limitam a autonomia e a mobilidade desse grupo, reforçando sua exclusão do convívio social.

Ademais, o preconceito velado e a desinformação continuam sendo barreiras simbólicas à inclusão. Muitas vezes, pessoas com deficiência são vistas sob a ótica da piedade ou da incapacidade, o que reforça estigmas e limita oportunidades, sobretudo no mercado de trabalho. Mesmo com incentivos legais para a contratação, muitas empresas não oferecem condições adequadas ou treinamentos para lidar com a diversidade, o que prejudica tanto a produtividade quanto o bem-estar dos profissionais.

A educação inclusiva, por sua vez, ainda enfrenta entraves estruturais e pedagógicos. Embora prevista em lei, a falta de professores capacitados e de recursos didáticos adaptados dificulta o aprendizado de alunos com deficiência, comprometendo seu desenvolvimento acadêmico e social.

Portanto, para que a sociedade brasileira avance rumo à verdadeira inclusão, é necessário um esforço conjunto entre poder público, iniciativa privada e sociedade civil. Investimentos em infraestrutura acessível, campanhas de conscientização e capacitação profissional são medidas urgentes. Além disso, é imprescindível que a inclusão deixe de ser um mero discurso político e se concretize em ações que valorizem a diversidade humana.

Afinal, uma sociedade justa e igualitária só é possível quando todos os seus cidadãos têm garantido o direito de viver com dignidade, autonomia e respeito às suas particularidades.
    """
    nota = avaliar_redacao(redacao_exemplo)
    print(f"Nota estimada: {nota}/1000")