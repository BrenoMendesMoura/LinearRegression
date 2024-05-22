# LinearRegression
LinearRegression



```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
df = pd.read_csv('sample_data/CarPrice_Assignment.csv', encoding='latin1')

colunas_nao_numericas = df.select_dtypes(include=['object']).columns

df = df.drop(columns=colunas_nao_numericas)

matriz_correlacao = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlacao, annot=True, cmap='coolwarm', center=0)
plt.title('Matriz de Correlação')
plt.show()
```

![image](https://github.com/BrenoMendesMoura/LinearRegression/assets/80074264/2ce26665-f0ba-4e80-a8d8-62426ba14257)

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

X = df[['price']].values
y = df['enginesize'].values

X_train, X_test, y_train, y_test = train_test_split(df[['price']].values, df['enginesize'].values, test_size=0.3, random_state=42)

modelo = LinearRegression()
modelo.fit(X_train, y_train)

y_pred = modelo.predict(X_test)
```
```python
plt.scatter(X_test, y_test, color='blue', label='Dados reais')
plt.plot(X_test, y_pred, color='red', label='Linha de regressão')
plt.xlabel('price')
plt.ylabel('enginesize')
plt.title('Regressão Linear: price vs. enginesize')
plt.legend()
plt.show()

print('Coeficiente angular (a):', modelo.coef_[0])
print('Intercepto (b):', modelo.intercept_)

## Prevendo a potência de acordo com o preço
# enginesize = (Coeficiente angular (a)) * preço + (Intercepto (b))

novo_preco = 100000
previsao_enginesize = modelo.predict([[novo_preco]])
print(f'Previsão de enginesize para o preço de {novo_preco}:', previsao_enginesize[0])
```


![image](https://github.com/BrenoMendesMoura/LinearRegression/assets/80074264/516c98cf-94e4-44bc-8f91-e9fa9409b409)
![image](https://github.com/BrenoMendesMoura/LinearRegression/assets/80074264/05bfc161-c163-4b7a-93bd-665384f59c50)


