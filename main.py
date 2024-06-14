import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar cada planilha em um DataFrame
produtos_df = pd.read_excel('produtos.xlsx')
clientes_df = pd.read_excel('clientes.xlsx')
pedidos_df = pd.read_excel('pedidos.xlsx')
itens_pedido_df = pd.read_excel('itens_pedido.xlsx')
estoque_df = pd.read_excel('estoque.xlsx')
categorias_df = pd.read_excel('categorias.xlsx')
fornecedores_df = pd.read_excel('fornecedores.xlsx')

# Tratar valores ausentes, outliers e dados inconsistentes
produtos_df = produtos_df.dropna()  # Remover linhas com valores ausentes

# Confirmar a remoção de linhas com valores ausentes
print("Produtos após remover valores ausentes:")
print(produtos_df)

# Padronizar formatos e unidades
produtos_df['preço'] = produtos_df['preço'].astype(float)  # Converter preços para o tipo float


# Confirmar a conversão de formato
print("\nFormato dos preços após conversão:")
print(produtos_df['preço'].dtype)


# Exibir as primeiras linhas de cada DataFrame
print("Produtos:")
print(produtos_df)

print("\nClientes:")
print(clientes_df)

print("\nPedidos:")
print(pedidos_df)

print("\nItens do Pedido:")
print(itens_pedido_df)

print("\nEstoque:")
print(estoque_df)

print("\nCategorias:")
print(categorias_df)

print("\nFornecedores:")
print(fornecedores_df)


# Classificação dos produtos com base no preço
produtos_df['classificação'] = produtos_df['preço'].apply(lambda x: 1 if x > 50 else 0)

# Valores verdadeiros (preços reais) e valores previstos (classificação baseada no preço)
y_true = produtos_df['classificação']
y_pred = produtos_df['classificação']

# Calcula a matriz de confusão
matriz_confusao = confusion_matrix(y_true, y_pred)

# Exibe a matriz de confusão
print("Matriz de Confusão:")
print(matriz_confusao)


# Dividir os dados em conjunto de treinamento e teste
X = produtos_df[['preço']]  # Features
y = produtos_df['classificação']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escolhe algoritmo de classificação
modelo = RandomForestClassifier()

# Treina o modelo
modelo.fit(X_train, y_train)

# Fazer previsões
y_pred = modelo.predict(X_test)

# Avaliar a precisão do modelo
precisao = accuracy_score(y_test, y_pred)
print("Precisão do modelo:", precisao)

# Identifica corretamente todos os exemplos positivos.
recall = recall_score(y_test, y_pred)
print("Recall do modelo:", recall)

# Definir os parâmetros para otimização
parametros = {'n_estimators': [50, 100, 150], 'max_depth': [None, 10, 20]}

# Criar o objeto GridSearchCV
grid_search = GridSearchCV(RandomForestClassifier(), parametros, cv=3)

# Executar a busca em grade
grid_search.fit(X_train, y_train)

# Melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)

# Avaliar o modelo com os melhores hiperparâmetros
melhor_modelo = grid_search.best_estimator_
y_pred = melhor_modelo.predict(X_test)

# Realizar validação cruzada
pontuacoes = cross_val_score(melhor_modelo, X, y, cv=3)

# Exibir as pontuações de validação cruzada
print("Pontuações de validação cruzada:", pontuacoes)

# Calcular a média das pontuações
media_pontuacoes = pontuacoes.mean()
print("Média das pontuações de validação cruzada:", media_pontuacoes)

#Estatísticas descritivas
media = produtos_df['preço'].mean()
mediana = produtos_df['preço'].median()
desvio_padrao = produtos_df['preço'].std()

print(f'Média: {media}, Mediana: {mediana}, Desvio Padrão: {desvio_padrao}')

# Gráfico de dispersão entre preço e quantidade de itens pedidos
plt.figure(figsize=(10, 6))
sns.scatterplot(x='preço_unitário', y='quantidade', data=itens_pedido_df)
plt.title('Relação entre Preço e Quantidade de Itens Pedidos')
plt.xlabel('Preço Unitario')
plt.ylabel('Quantidade')
plt.show()

# Identificação de padrões nos dados
# Exemplo: Contagem de produtos por categoria
tendencia_categoria = produtos_df['nome'].value_counts()
print('Contagem de Produtos por Categoria:')
print(tendencia_categoria)

# ENTREGA 2 CIENCIAS DE DADOS - Modelagem Estatística

# Calculando o preço total do pedido (variável dependente)
itens_pedido_df['preco_total'] = itens_pedido_df['quantidade'] * itens_pedido_df['preço_unitário']

# Separando as variáveis independentes (X) e a variável dependente (y)
X = itens_pedido_df[['quantidade', 'preço_unitário']]  # Variáveis independentes
y = itens_pedido_df['preco_total']  # Variável dependente

# Adicionando uma constante (intercepto) ao modelo
X = sm.add_constant(X)

# Ajustando o modelo de regressão linear
modelo = sm.OLS(y, X)
resultados = modelo.fit()

# Exibindo um resumo dos resultados da regressão
print(resultados.summary())

# Inicializar o modelo de regressão linear
modelo = LinearRegression()

# Avaliar modelo
R2 = resultados.rsquared
print(f'R-squared: {R2}')

# Definir as características dos novos itens do pedido
X_treino = itens_pedido_df[['quantidade', 'preço_unitário']]
y_treino = itens_pedido_df['id_produto']  # Suponha que 'id_produto' seja o alvo do modelo

# Criar e treinar o modelo de regressão linear
modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

# Definir as características dos novos itens do pedido
X_novos_dados = itens_pedido_df[['quantidade', 'preço_unitário']]

# Fazer previsões para os novos dados
previsoes = modelo.predict(X_novos_dados)

# Exibir as previsões
print("Previsões para os novos itens do pedido:")
print(previsoes)

# Avaliar desempenho
y_verdadeiro = itens_pedido_df['id_produto']  # Suponha que 'id_produto' seja o valor verdadeiro

# Calcular o erro médio quadrático
erro_medio_quadratico = mean_squared_error(y_verdadeiro, previsoes)
print(f'Erro Médio Quadrático: {erro_medio_quadratico}')

# Preparar os dados para os modelos
X1 = itens_pedido_df[['quantidade', 'preço_unitário']]
X2 = itens_pedido_df[['quantidade', 'id_produto']]
y = itens_pedido_df['id']

# Instanciar os modelos de regressão linear
modelo1 = LinearRegression()
modelo2 = LinearRegression()

# Ajustar os modelos aos dados
modelo1.fit(X1, y)
modelo2.fit(X2, y)

# Fazer previsões
previsoes1 = modelo1.predict(X1)
previsoes2 = modelo2.predict(X2)

# Avaliar os modelos
erro_medio_quadratico1 = mean_squared_error(y, previsoes1)
erro_medio_quadratico2 = mean_squared_error(y, previsoes2)

# Comparar modelos
if erro_medio_quadratico1 < erro_medio_quadratico2:
    print('Modelo 1 é melhor.')
else:
    print('Modelo 2 é melhor.')





