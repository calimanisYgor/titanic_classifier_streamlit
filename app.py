import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Função para carregar os dados
@st.cache_data
def load_data():
    path = "titanic.csv"
    header = ["passengerid", "survived", "pclass", "name", "sex", "age", "sibsp", "parch", "ticket", "fare", "cabin", "embarked"]
    df = pd.read_csv(path, header=0, names=header)
    return df

# Função para exibir informações básicas sobre os dados
def display_data(df):
    st.write("Primeiras linhas do dataframe:")
    st.dataframe(df.head())
    
    st.write("Últimas linhas do dataframe:")
    st.dataframe(df.tail())
    
    st.write("Informações sobre o dataframe:")
    st.text(df.info())

# Função para plotar gráfico de barras da contagem de sobreviventes
def plot_survivor_count(df):
    colors = {'Sobreviventes': 'blue', 'Mortos': 'red'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    
    plt.figure(figsize=(10, 6))
    plt.legend(handles, labels, title='Legenda')
    sns.countplot(x='survived', data=df, palette={'0': 'red', '1': 'blue'})
    plt.title('Contagem de sobreviventes do Titanic')
    st.pyplot(plt)

# Função para treinar o modelo KNN
def train_model(df):
    entradas = df[['survived', 'pclass']]
    classes = df['sex']
    
    entradas_treino, entradas_teste, classes_treino, classes_teste = train_test_split(entradas, classes, test_size=0.2)

    k = st.slider('Selecione o valor de K para o KNN', 1, 20, 7)
    modelo = KNeighborsClassifier(n_neighbors=k)
    modelo.fit(entradas_treino, classes_treino)
    
    classes_encontradas = modelo.predict(entradas_teste)
    acertos = accuracy_score(classes_teste, classes_encontradas)
    
    st.write(f"Acerto médio de classificação com K={k}: {acertos:.2f}")
    
    return modelo

# Função para fazer predição de sobrevivência
def predict_survival(modelo):
    st.write("Previsão de sobrevivência")
    survived = st.number_input("Insira a probabilidade de sobrevivência (0 ou 1):", 0, 1)
    pclass = st.selectbox("Selecione a classe:", [1, 2, 3])
    
    previsao = modelo.predict([[survived, pclass]])
    st.write(f"A previsão do modelo é: {'Mulher' if previsao[0] == 'female' else 'Homem'}")

# Função principal
def main():
    st.title("Titanic Survival Prediction ")
    
    # Carregando os dados
    df = load_data()
    
    # Exibindo os dados
    if st.checkbox("Mostrar dados do Titanic"):
        display_data(df)
    
    # Gráfico de sobreviventes
    if st.checkbox("Mostrar gráfico de sobreviventes"):
        plot_survivor_count(df)
    
    # Treinar o modelo
    if st.checkbox("Treinar modelo KNN"):
        modelo = train_model(df)
    
        # Fazer predições
        if st.checkbox("Fazer predição com o modelo treinado"):
            predict_survival(modelo)

# Executando o aplicativo
if __name__ == "__main__":
    main()
