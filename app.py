import streamlit as st
import pandas as pd
import nltk
import string
import re
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import random
import matplotlib.pyplot as plt
from collections import Counter

# Downloads necessários (só executa na primeira vez)
nltk.download('punkt')
nltk.download('stopwords')

# Vocabulário de sentimentos
positivas = {"ótimo", "excelente", "adorei", "bom", "maravilhoso", "superou", "perfeito", "gostei", "satisfatória", "qualidade", "resolução", "bonita"}
negativas  = {"ruim", "horrível", "péssimo", "não gostei", "esperava mais", "lento", "defeito", "problema"}
neutras    = {"regular", "ok", "aceitável", "mediano", "normal", "cumpre"}

# Funções auxiliares para pré-processamento
def preprocess(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('portuguese'))
    tokens = [t for t in tokens if t not in string.punctuation and t not in stop_words]
    return tokens

def build_features(tokens):
    return {word: True for word in tokens}

# Função de rotulagem automática
def rotular_automaticamente(texto):
    texto_limpo = re.sub(r'[^\w\s]', '', texto.lower())
    if any(p in texto_limpo for p in positivas):
        return "positivo"
    elif any(n in texto_limpo for n in negativas):
        return "negativo"
    elif any(nu in texto_limpo for nu in neutras):
        return "neutro"
    else:
        return "neutro"

# Carrega e prepara o dataset e treina o modelo
@st.cache_data
def treinar_modelo(df):
    df['sentimento'] = df['Reviews'].apply(rotular_automaticamente)
    dataset = [(build_features(preprocess(row['Reviews'])), row['sentimento']) for _, row in df.iterrows()]
    random.shuffle(dataset)
    split = int(0.8 * len(dataset))
    train_set = dataset[:split]
    return NaiveBayesClassifier.train(train_set)

# Interface Streamlit
st.set_page_config(page_title="Sentimento AI - CSV & ML", layout="wide")
st.title("📊 Sentimento AI: Análise de Reviews")
st.markdown("Classificador de sentimentos com aprendizado de máquina e visualização de estatísticas.")

# Carregar o CSV via file uploader
uploaded_file = st.file_uploader("Carregue o arquivo CSV", type=["csv"])
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file, sep=';', encoding='utf-8')
    except:
        df = pd.read_csv(uploaded_file, sep=';', encoding='latin1')
    
    # Verifica se a coluna 'Reviews' existe
    if 'Reviews' not in df.columns:
        st.error("O CSV deve conter a coluna 'Reviews'.")
        st.stop()

    # Treinar o modelo com os dados carregados
    modelo = treinar_modelo(df)
    
    # Calcular as estatísticas de sentimento
    sentimentos_csv = df['Reviews'].dropna().apply(rotular_automaticamente)
    contagem = Counter(sentimentos_csv)

    # Exibe o gráfico de estatísticas na sidebar
    if contagem:
        st.sidebar.header("📊 Estatísticas no Dataset CSV")
        labels, values = zip(*contagem.items())
        fig, ax = plt.subplots()
        cores = ["#28a745" if s == "positivo" else "#dc3545" if s == "negativo" else "#ffc107" for s in labels]
        ax.bar(labels, values, color=cores)
        ax.set_title("Distribuição de Sentimentos")
        st.sidebar.pyplot(fig)

    # Se existir a coluna 'produto', permite seleção; caso contrário, usa Produto Genérico
    if "produto" in df.columns:
        produto_selecionado = st.selectbox("📱 Selecione um produto:", df["produto"].unique())
        df_estat = df[df["produto"] == produto_selecionado]
    else:
        df["produto"] = "Produto Genérico"
        produto_selecionado = "Produto Genérico"
        df_estat = df

    st.markdown(f"### ✏️ Avaliações para: `{produto_selecionado}`")
    for _, row in df_estat.iterrows():
        frase = row['Reviews']
        sentimento = rotular_automaticamente(frase)
        emoji = {"positivo": "🙂", "negativo": "😞", "neutro": "😐"}.get(sentimento, "❓")
        cor = "#DFF6DD" if sentimento == "positivo" else "#F8D7DA" if sentimento == "negativo" else "#FFF3CD"
        st.markdown(f"""
            <div style='padding: 12px; border-radius: 8px; background-color: {cor}; margin-bottom: 10px;'>
                <strong style='font-size: 18px;'>{emoji} {sentimento.capitalize()}</strong><br>
                <span style='font-size: 15px;'>{frase}</span>
            </div>
        """, unsafe_allow_html=True)
else:
    st.info("Carregue o arquivo CSV para prosseguir.")
