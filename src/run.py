# Previsão do custo pessoal médico
# O objetivo desse projeto é testar diferentes regressões para a mesma base de dados. 
# O intuito é visualizar as funcionalidades do ML Flow como ferramenta de tracking de experimentos de machine learning.

import os
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

from data_pipeline import transform_data, split_data
from ml_flow_functions import MlFlowLab_Regressor

# Caminhos do projeto
rootPath = os.getcwd()
dataPath = os.path.join(rootPath, 'data')

# Importando os dados
dados = pd.read_csv(os.path.join(dataPath, 'insurance.csv'))

# Preparando dados para treinamento
df = transform_data(dados)
X_train, X_test, y_train, y_test = split_data(df, "charges", 0.3)

# Realizando experimento

MlFlowLab_Regressor(
    nome_experimento="Previsão de custos médicos pessoais",
    modelo=GradientBoostingRegressor(random_state=1337), # Poderia ser qualquer outro dos modelos importados
    X_train=X_train, y_train=y_train,
    X_test=X_test, y_test=y_test)\
        .new_run(
            "Gradient Boosting"
        )