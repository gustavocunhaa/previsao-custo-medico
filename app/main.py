# Endpoint Flask que carrega o melhor modelo disponibilizado via MlFlow
from flask import Flask, request
import mlflow
import pandas as pd

app = Flask(__name__)

# Modelo carregado como uma PyFuncModel.
logged_model = 'runs:/c11d0a7d5ea84fcda7c080694c51d2fa/GradientBoostingRegressor(random_state=1337)'
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Dados que vão ser recebidos pelo endpoint
coluna_dados = ['age', 'sex', 'children', 'smoker']

@app.route("/")
def inicial():
    return "Caminho raiz, use /predict/ e faça o input dos dados para predizer a estrutura"

@app.route("/predict/", methods=['POST'])
def preditct():

    dados = request.get_json()
    dados_input = [dados[col] for col in coluna_dados]
    df_predicao = pd.DataFrame(dados_input)
   
    try:
        predict = loaded_model.predict(pd.DataFrame(df_predicao))
        response = {200, predict}

    except Exception as e:
        response = {500, e} 
    
    return response

if __name__ == "__main__":
    app.run(debug=True)