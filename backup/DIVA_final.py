# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 13:43:24 2024

@author: Miguel
"""

#%%Importação de bibliotecas
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

#%%carregamento da base
df = pd.read_excel('Dados do projeto DIVA.xlsx')

#%%Pre-processamento

df.drop(columns = ['Participante'], inplace = True) #remove a coluna participantes

from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

# Ativando o novo comportamento
pd.set_option('future.no_silent_downcasting', True)

#substitui os valores no df
df.replace('S',1,inplace=True)
df.replace('N',0,inplace=True)
df.replace('s',1,inplace=True)
df.replace('n',0,inplace=True)
#df.replace(np.nan,0,inplace=True) #0 é a moda
df.replace('FEM',1,inplace=True)
df.replace('fem',1,inplace=True)
df.replace('MAS',0,inplace=True)
df.replace('MASC',0,inplace=True)
df.replace('mas',0,inplace=True)
df.replace('masc',0,inplace=True)

df['Altura'] = df['Altura'].str.replace(',', '.').astype(float)

for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

#Para o pré-processamento foi utilizado métodos simples pois a base
#acessada era composta de poucos elementos e com isso
#tinha pouca coisa a ser mudada
y = df['Acesso venoso difícil']
print(y.value_counts())
X = df.drop(columns=['Acesso venoso difícil'])

# Contando valores ausentes em cada coluna
imputer = KNNImputer(n_neighbors=1)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

#%% Defina os modelos que você deseja testar
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, SMOTEN, ADASYN, SVMSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline
from imblearn.base import BaseSampler

# Dados de exemplo
# X, y são seus dados de características e rótulos
# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 2: Definir os modelos e parâmetros
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Decision Tree Balanced': DecisionTreeClassifier(class_weight='balanced'),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(max_iter=1000)
}

parameters = {
    'Decision Tree': {'clf__max_depth': [None, 2, 3, 4]},
    'Decision Tree Balanced': {'clf__max_depth': [None, 2, 3, 4]},
    'SVM': {'clf__C': [0.1, 1, 10], 'clf__gamma': [0.1, 0.01, 0.001]},
    'Logistic Regression': {'clf__C': [0.1, 1, 10]}
}

# Passo 3: Iterar sobre os modelos e parâmetros para encontrar o melhor modelo
best_model = None
best_score = 0

for model_name, model in models.items():
    # Criando o pipeline com SMOTE e o modelo
    pipeline =  ImbPipeline([
#        ('sampler', SMOTE(random_state=42)),
        ('clf', model)
    ])
    
    # Criando o GridSearchCV com o pipeline
    clf = GridSearchCV(pipeline, parameters[model_name], scoring='balanced_accuracy', verbose=3, cv = 10)
    clf.fit(X_train, y_train)
    
    score = clf.best_score_
    print(f"{model_name}: {score}")
    
    if score > best_score:
        best_model = clf.best_estimator_
        best_score = score

# O melhor modelo e seus parâmetros
print("Melhor modelo:", best_model)

# Faça previsões nos dados de teste
y_pred = best_model.predict(X_test)
report = classification_report(y_test, y_pred)
print(report)
