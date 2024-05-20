# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:43:24 2024

@author: Miguel
"""

#%%Importação de bibliotecas
# Core libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot

# Sklearn functionality
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold

#%%carregamento da base

df = pd.read_excel('Dados do projeto DIVA.xlsx')

#%%Pre-processamento

df.drop(columns = ['Participante'], inplace = True) #remove a coluna participantes

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

#substitui os valores no df
df.replace('S',1,inplace=True)
df.replace('N',0,inplace=True)
df.replace('s',1,inplace=True)
df.replace('n',0,inplace=True)
df.replace(np.nan,0,inplace=True) #0 é a moda
df.replace('FEM',1,inplace=True)
df.replace('fem',1,inplace=True)
df.replace('MAS',0,inplace=True)
df.replace('MASC',0,inplace=True)
df.replace('mas',0,inplace=True)
df.replace('masc',0,inplace=True)

#Para o pré-processamento foi utilizado métodos simples pois a base
#acessada era composta de poucos elementos e com isso
#tinha pouca coisa a ser mudada
y = df['Acesso venoso difícil']
X = df.drop(columns=['Acesso venoso difícil'])

#%%Treinamento

# Build a decision tree model
model = DecisionTreeClassifier(class_weight='balanced')

# Definindo o grid de hiperparâmetros para serem testados
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Criando o objeto GridSearchCV para a Nested Cross-Validation
inner_cv = KFold(n_splits = 5, shuffle=True, random_state = 42)
outer_cv = KFold(n_splits = 5, shuffle=True, random_state = 42)

# Criando o objeto GridSearchCV
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, cv = inner_cv)

# Treinando o GridSearchCV
grid_search.fit(X, y)

# Realizando a Nested Cross-Validation
nested_score = cross_val_score(grid_search, X = X, y = y, cv = outer_cv)
# Imprimindo os resultados
print("Acurácia Média: %0.2f (+/- %0.2f)" % (nested_score.mean(), nested_score.std() * 2))

# Imprimindo os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros encontrados:")
print(grid_search.best_params_)
