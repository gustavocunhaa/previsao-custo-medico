import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 1337

def transform_data(dados: pd.DataFrame):
    # Fazendo o enconding das variáveis binárias
    map_sex = {'female': 1, 'male': 0}
    map_smoker = {'yes': 1, 'no': 0}
    dados['sex'] = dados['sex'].replace(map_sex)
    dados['smoker'] = dados['smoker'].replace(map_smoker)
    
    # Removendo coluna não utilizada
    df = dados.drop(columns=['region'])    
    return df

def split_data(df: pd.DataFrame, target: str, test_size: float):
    # Separando variáveis explicativas da target
    x = df.drop(columns=[target])
    y = df[target]

    # Separação da base de dados em treino e teste para cada normalização
    X_train, X_test, y_train, y_test = train_test_split(x, y, 
                                                        test_size=test_size, 
                                                        random_state=SEED)
    
    return X_train, X_test, y_train, y_test




