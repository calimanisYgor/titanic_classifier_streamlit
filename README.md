# Titanic Survival Prediction Dashboard

Este é um mini projeto que utiliza o dataset do Titanic para realizar predições de sobrevivência utilizando o algoritmo K-Nearest Neighbors (KNN). O projeto foi desenvolvido utilizando Streamlit para criar um dashboard interativo.

## Requisitos

- Python 3.7 ou superior
- [Pandas](https://pandas.pydata.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Streamlit](https://streamlit.io/)
- [Scikit-learn](https://scikit-learn.org/)

## Instalação

1. Clone este repositório para o seu ambiente local:

    ```bash
    git clone https://github.com/usuario/titanic-knn-dashboard.git
    ```

2. Navegue até o diretório do projeto:

    ```bash
    cd titanic-knn-dashboard
    ```

3. Crie um ambiente virtual (venv):

    ```bash
    python -m venv venv
    ```

4. Ative o ambiente virtual:

    - No Windows:
      ```bash
      venv\Scripts\activate
      ```
    - No macOS/Linux:
      ```bash
      source venv/bin/activate
      ```

5. Instale as dependências necessárias:

    ```bash
    pip install -r requirements.txt
    ```

## Como Executar

Para rodar o dashboard interativo, utilize o comando abaixo:

```bash
streamlit run app.py
