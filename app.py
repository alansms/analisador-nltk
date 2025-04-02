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

# Downloads necessários
nltk.download('punkt')
nltk.download('stopwords')

# Variáveis globais de vocabulário
positivas = {"ótimo", "excelente", "adorei", "bom", "maravilhoso", "superou", "perfeito", "gostei", "satisfatória", "qualidade", "resolução", "bonita"}
negativas = {"ruim", "horrível", "péssimo", "não gostei", "esperava mais", "lento", "defeito", "problema"}
neutras = {"regular", "ok", "aceitável", "mediano", "normal", "cumpre"}

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

# Carrega e prepara o dataset rotulado automaticamente
@st.cache_data
def treinar_modelo():
    try:
        df = pd.read_csv("/mnt/data/reviews_smartphone.csv", sep=';', encoding='utf-8')
    except:
        df = pd.read_csv("/mnt/data/reviews_smartphone.csv", sep=';', encoding='latin1')

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

modelo = treinar_modelo()

# Carregar CSV e mostrar gráfico com base no conteúdo
try:
    df_estat = pd.read_csv("/mnt/data/reviews_smartphone.csv", sep=';', encoding='utf-8')
except:
    df_estat = pd.read_csv("/mnt/data/reviews_smartphone.csv", sep=';', encoding='latin1')

if 'Reviews' in df_estat.columns:
    sentimentos_csv = df_estat['Reviews'].dropna().apply(rotular_automaticamente)
    contagem = Counter(sentimentos_csv)

    if contagem:
        st.sidebar.header("📊 Estatísticas no Dataset CSV")
        labels, values = zip(*contagem.items())
        fig, ax = plt.subplots()
        cores = ["#28a745" if s == "positivo" else "#dc3545" if s == "negativo" else "#ffc107" for s in labels]
        ax.bar(labels, values, color=cores)
        ax.set_title("Distribuição de Sentimentos")
        st.sidebar.pyplot(fig)

    produto_selecionado = st.selectbox("📱 Selecione um produto:", df_estat["produto"].unique() if "produto" in df_estat.columns else ["Produto Genérico"])
    if "produto" not in df_estat.columns:
        df_estat["produto"] = "Produto Genérico"

    reviews = df_estat[df_estat["produto"] == produto_selecionado]
    st.markdown(f"### ✏️ Avaliações para: `{produto_selecionado}`")

    for _, row in reviews.iterrows():
        frase = row['Reviews']
        sentimento, emoji = rotular_automaticamente(frase), {"positivo": "🙂", "negativo": "😞", "neutro": "😐"}.get(rotular_automaticamente(frase), "❓")
        cor = "#DFF6DD" if sentimento == "positivo" else "#F8D7DA" if sentimento == "negativo" else "#FFF3CD"

        st.markdown(f"""
            <div style='padding: 12px; border-radius: 8px; background-color: {cor}; margin-bottom: 10px;'>
                <strong style='font-size: 18px;'>{emoji} {sentimento.capitalize()}</strong><br>
                <span style='font-size: 15px;'>{frase}</span>
            </div>
        """, unsafe_allow_html=True)
else:
    st.error("❗ O CSV deve conter ao menos a coluna 'Reviews'.")
