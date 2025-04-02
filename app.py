import streamlit as st
import re
import random
import pandas as pd
import io

# Dicionário de palavras positivas, negativas e emoções específicas
palavras_positivas = {"bom", "ótimo", "excelente", "maravilhoso", "feliz", "alegria", "positivo", "sucesso", "incrível",
                      "fantástico", "adorável"}
palavras_negativas = {"ruim", "péssimo", "horrível", "triste", "fracasso", "negativo", "chato", "desastroso",
                      "deprimente", "lamentável"}
palavras_raiva = {"ódio", "raiva", "furioso", "irritado", "revoltado", "explosivo", "agressivo", "furibundo"}
palavras_medo = {"medo", "assustado", "pavor", "ameaça", "desesperado"}

# Mensagens motivacionais para sentimentos negativos
mensagens_apoio = [
    "💙 Respire fundo. Você não está sozinho. 💙",
    "🌿 Tente ouvir uma música relaxante e cuidar de você. 🌿",
    "🌟 Lembre-se: dias difíceis passam. Você é mais forte do que imagina. 🌟"
]

# Configuração de cores para realçar emoções
cores = {
    "Muito Positivo": "#DFF6DD",  # Verde claro
    "Positivo": "#C3E6CB",
    "Neutro": "#FFF3CD",  # Amarelo claro
    "Negativo": "#F8D7DA",  # Vermelho claro
    "Muito Negativo": "#F5C6CB",
    "Raiva": "#FF5733",  # Laranja escuro
    "Medo": "#6A5ACD"  # Azul escuro
}

# Função de análise de sentimento

def analisar_sentimento(frase):
    if not frase.strip():
        return "Por favor, insira uma frase válida.", "🧐"

    frase = re.sub(r'[^\w\s]', '', frase.lower())
    palavras = frase.split()

    contagem_positiva = sum(1 for palavra in palavras if palavra in palavras_positivas)
    contagem_negativa = sum(1 for palavra in palavras if palavra in palavras_negativas)
    contagem_raiva = sum(1 for palavra in palavras if palavra in palavras_raiva)
    contagem_medo = sum(1 for palavra in palavras if palavra in palavras_medo)

    if contagem_raiva > 0:
        return "Raiva", "🔥"
    elif contagem_medo > 0:
        return "Medo", "😨"
    elif contagem_positiva >= 3:
        return "Muito Positivo", "😍"
    elif contagem_positiva > contagem_negativa:
        return "Positivo", "🙂"
    elif contagem_negativa >= 3:
        return "Muito Negativo", "😢"
    elif contagem_negativa > contagem_positiva:
        return "Negativo", "😞"
    else:
        return "Neutro", "😐"

# Interface Web com Streamlit
st.set_page_config(page_title="Sentimento AI - Análise de Reviews", page_icon="📱", layout="wide")

st.markdown("""
    <style>
        .big-title {
            font-size: 42px;
            text-align: center;
            color: #4A90E2;
        }
        .sub-title {
            font-size: 22px;
            text-align: center;
            color: #666;
        }
        .result-box {
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            width: 60%;
            margin: auto;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='big-title'>📱 Sentimento AI: Análise de Reviews</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Visualize os sentimentos atribuídos às avaliações de smartphones</p>", unsafe_allow_html=True)

# Carregamento do CSV
uploaded_file = st.sidebar.file_uploader("📄 Faça upload do arquivo de reviews (CSV)", type="csv")

if uploaded_file:
    try:
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio, sep=';')
    except UnicodeDecodeError:
        stringio = io.StringIO(uploaded_file.getvalue().decode("latin1"))
        df = pd.read_csv(stringio, sep=';')
    except Exception as e:
        st.error(f"Erro ao processar o CSV: {e}")
        st.stop()

    st.success("Arquivo carregado com sucesso!")

    # Normaliza os nomes das colunas
    df.columns = [col.strip().lower() for col in df.columns]

    # Tenta renomear colunas parecidas com 'produto' e 'review'
    colunas_renomeadas = {}
    for col in df.columns:
        if 'produto' in col and col != 'produto':
            colunas_renomeadas[col] = 'produto'
        if 'review' in col and col != 'review':
            colunas_renomeadas[col] = 'review'
    df.rename(columns=colunas_renomeadas, inplace=True)

    st.write("🔍 Colunas detectadas:", df.columns.tolist())

    # Se a coluna 'produto' estiver ausente, adiciona uma com valor padrão
    if 'produto' not in df.columns:
        df['produto'] = 'Produto Genérico'
        st.warning("Coluna 'produto' não encontrada. Adicionada automaticamente com valor padrão.")

    if "produto" in df.columns and "review" in df.columns:
        produto_selecionado = st.selectbox("📱 Selecione um produto:", df["produto"].unique())
        reviews_produto = df[df["produto"] == produto_selecionado]

        st.markdown(f"### 📝 Avaliações para: `{produto_selecionado}`")

        for i, row in reviews_produto.iterrows():
            texto = row['review']
            sentimento, emoji = analisar_sentimento(texto)
            cor = cores.get(sentimento, "#ffffff")

            st.markdown(f"""
            <div class='result-box' style='background-color: {cor};
                 color: {'#000' if sentimento in ['Muito Positivo', 'Neutro'] else '#fff'};'>
                {emoji} {sentimento}<br><span style='font-size: 16px; font-weight: normal;'>{texto}</span>
            </div><br>
            """, unsafe_allow_html=True)

            if "Negativo" in sentimento or sentimento in ["Raiva", "Medo"]:
                st.info(random.choice(mensagens_apoio))
    else:
        st.error("⚠️ O CSV precisa conter as colunas 'produto' e 'review'.")
else:
    st.info("👈 Faça upload de um arquivo CSV contendo avaliações para visualizar os sentimentos.")
