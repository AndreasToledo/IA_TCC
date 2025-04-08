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
    A inteligência artificial (IA) é uma das tecnologias mais disruptivas da atualidade, com impactos profundos em várias áreas da sociedade. De assistentes virtuais a carros autônomos, a IA está mudando rapidamente a forma como vivemos e trabalhamos. No entanto, seu avanço também levanta questões éticas e sociais que precisam ser discutidas e regulamentadas, a fim de garantir que seus benefícios sejam amplamente distribuídos sem prejudicar direitos fundamentais. No campo da saúde, a IA tem sido utilizada para diagnosticar doenças com precisão, muitas vezes superando os métodos tradicionais. Isso tem permitido tratamentos mais rápidos e eficientes, o que pode salvar muitas vidas. No entanto, a automação de profissões como o atendimento médico e o atendimento ao cliente também gera preocupações sobre o desemprego e a precarização do trabalho. Outro desafio relacionado à IA é a privacidade. A coleta massiva de dados para treinar algoritmos levanta sérias questões sobre a proteção das informações pessoais. É fundamental que a legislação acompanhe o desenvolvimento da tecnologia, garantindo que os dados dos cidadãos sejam tratados com respeito e segurança. Em síntese, a inteligência artificial é uma ferramenta poderosa para o progresso da sociedade, mas seu uso deve ser regulamentado para evitar danos sociais e garantir que seus benefícios sejam distribuídos de forma justa. O governo, as empresas e a sociedade precisam trabalhar juntos para criar um ambiente de inovação responsável e ético.
    """
    nota = avaliar_redacao(redacao_exemplo)
    print(f"Nota estimada: {nota}/1000")
