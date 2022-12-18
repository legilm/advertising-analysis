import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# - Passo 1: Entendimento do Desafio
# - Passo 2: Entendimento da Área/Empresa
# - Passo 3: Extração/Obtenção de Dados
# - Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# - Passo 5: Análise Exploratória
# - Passo 6: Modelagem + Algoritmos (Aqui que entra a Inteligência Artificial, se necessário)
# - Passo 7: Interpretação de Resultados

# import database

table = pd.read_csv("advertising.csv")
print(table)

# View database and made the adjustments
# make the correlations for each media
# matplotlib and seaborn
# Create graph

sns.heatmap(table.corr(), annot=True, cmap="Wistia")

# Show graph

plt.show()

# # Exploratory analysis -> understand how the base is working
# Isolate training data and test data
# y -> Data to be forecasted
# x -> Data to be use to training

y = table["Vendas"]
x = table[["TV", "Radio", "Jornal"]]
x_training, x_test, y_training, y_test = train_test_split(x,y, test_size=0.3)

# # Build the artifical inteligence to made the forecast
# Define the method

model_linear_regression = LinearRegression()
model_random_forest_regressor = RandomForestRegressor()

# Train each AI

model_linear_regression.fit(x_training, y_training)
model_random_forest_regressor.fit(x_training, y_training)

# test the each AI

forecast_linear_regression = model_linear_regression.predict(x_test)
forecast_random_forest_regressor = model_random_forest_regressor.predict(x_test)

# Utilize the R² to define the best one

percentage_linear_regression = r2_score(y_test, forecast_linear_regression) * 100
percentage_random_forest_regressorprint = r2_score(y_test, forecast_random_forest_regressor) * 100

print(f"Regressão Linear: {percentage_linear_regression}%")
print(f"Árvore de Decisão: {percentage_random_forest_regressorprint}%")

if percentage_linear_regression < percentage_random_forest_regressorprint:
    print("O método que chegou mais próximo do valor correto foi o de Árvore de decisão")
else:
    print("O método que chegou mais próximo do valor correto foi o de Regressão linear")

# Made a forecast for investment in the next month, based in the values in the "novos.csv" file
novos = pd.read_csv("novos.csv")
print(novos)
if percentage_linear_regression < percentage_random_forest_regressorprint:
    forecast_next_month = model_random_forest_regressor.predict(novos)
    print("Utlizando o modelo de Árvore de Decisão para calcular a previsão para o próximo mês")
else:
    forecast_next_month = model_linear_regression.predict(novos)
    print("Utlizando o modelo de Regressão linear para calcular a previsão para o próximo mês")

print(forecast_next_month)
