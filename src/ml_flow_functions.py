import mlflow
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


# Construindo uma classe que irá rodar todo o experimento no mlflow
# A ideia é que ele receba os parametros, como o nome do experimento, o modelo e os dados
# Irá treinar um regressor e validar ele com base nas métricas setadas

class MlFlowLab_Regressor():

    def __init__(self, 
                 nome_experimento, 
                 modelo,
                 X_train, y_train, X_test, y_test):
        
        self.modelo = modelo
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        self.nome_experimento = nome_experimento
        self.id_experimento = mlflow.set_experiment(nome_experimento).experiment_id
        
        print(f"Experiment_id: {self.id_experimento}")

    def train_model(self):
        modelo = self.modelo

        modelo_treinado = modelo.fit(self.X_train, self.y_train)
        mlflow.sklearn.log_model(modelo_treinado, str(modelo)) # Seta o log do modelo
        
        return modelo_treinado

    def predict(self, modelo_treinado):
        predictions = modelo_treinado.predict(self.X_test)
        return predictions

    def evalute_model(self, predictions):
        y_test = self.y_test
        
        mse = mean_squared_error(y_test, predictions)
        mlflow.log_metric('mse',mse)

        rmse =  np.sqrt(mse)
        mlflow.log_metric('rmse',rmse)

        r2 = r2_score(y_test, predictions)
        mlflow.log_metric('r2',r2)

        json_results = {
            "mse": mse,
            "rmse": rmse,
            "r2": r2
        }
        
        return json_results


    def new_run(self, run_name):

        with mlflow.start_run(experiment_id=self.id_experimento):

            mlflow.set_tag("mlflow.runName", run_name)
            modelo_treinado = self.train_model()
            predictions = self.predict(modelo_treinado)
            json_results = self.evalute_model(predictions)

        return print(f"Experimento: {self.id_experimento} | Run: {run_name}. \nMétricas \n {json_results}")