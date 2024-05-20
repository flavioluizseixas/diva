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
print("Missing values: ", X.isna().sum())
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
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Dados de exemplo
# X, y são seus dados de características e rótulos
# Dividindo os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Passo 2: Definir os modelos e parâmetros
models = {
    'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
    'Random Forest': RandomForestClassifier(class_weight='balanced'),
    'SVM': SVC(class_weight='balanced'),
    'Logistic Regression': LogisticRegression(class_weight='balanced')
}

parameters = {
    'Decision Tree': {'clf__max_depth': [None, 2, 3, 4]},
    'Random Forest': {'clf__n_estimators': [50, 100, 200]},
    'SVM': {'clf__C': [0.1, 1, 10], 'clf__gamma': [0.1, 0.01, 0.001]},
    'Logistic Regression': {'clf__C': [0.1, 1, 10]}
}

# Passo 3: Iterar sobre os modelos e parâmetros para encontrar o melhor modelo
best_model = None
best_score = 0

for model_name, model in models.items():
    # Criando o pipeline com SMOTE e o modelo
    pipeline = ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])
    
    # Criando o GridSearchCV com o pipeline
    clf = GridSearchCV(pipeline, parameters[model_name], cv=5)
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



#%%SMOTE





import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from imblearn.over_sampling import BorderlineSMOTE, SMOTE, SMOTEN, ADASYN, SVMSMOTE

#Técnicas de smore escolhidas
sm = SMOTE(random_state=42)
bsm = BorderlineSMOTE(random_state=42)
smn = SMOTEN(random_state=42)
ada = ADASYN(random_state=42)
svms = SVMSMOTE(random_state=42)

X_train1, y_train1 = sm.fit_resample(X_train, y_train)
X_train2, y_train2 = bsm.fit_resample(X_train, y_train)
X_train3, y_train3 = smn.fit_resample(X_train, y_train)
X_train4, y_train4 = ada.fit_resample(X_train, y_train)
X_train5, y_train5 = svms.fit_resample(X_train, y_train)


#%%Cross-validate
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

#scores escolhidos
scoring = ['accuracy','precision_macro', 'recall_macro','f1_macro']


#função para calcular os resultados
def result_tree (X_train, y_train):
    
    #100 rodagens de 10 folds = 100*10 = 1000 execuções
    for pos in range(0,100): 
        tree1 = DecisionTreeClassifier(class_weight='balanced',max_depth=4)
        scores = cross_validate(tree1,X_train,y_train, scoring=scoring,cv=10)
        
        #média dos 10 folds
        acc = scores['test_accuracy'].mean()
        prec = scores['test_precision_macro'].mean()
        rec = scores['test_recall_macro'].mean()
        f1 = scores['test_f1_macro'].mean()
        
        if pos == 0:
            result = np.array([acc,prec,rec,f1])
        else:        
            result = np.vstack([result,np.array([acc,prec,rec,f1])])

    return np.mean(result,axis=0)

#roda para ver qual técnica de SMOTE teve o melhor resultado
#no caso a terceira técnica foi melhor
result1 = result_tree(X_train1, y_train1)
result2 = result_tree(X_train2, y_train2)
result3 = result_tree(X_train3, y_train3)
result4 = result_tree(X_train4, y_train4)
result5 = result_tree(X_train5, y_train5)


#%%Treinamento e matriz de confusao
#treinamento
tree1 = DecisionTreeClassifier(class_weight='balanced',max_depth=4)
tree1.fit(X_train3,y_train3)
y_pred = tree1.predict(X_test)
print(classification_report(y_test, y_pred))

#matriz confusão
from yellowbrick.classifier import ConfusionMatrix
cm = ConfusionMatrix(
    tree1,
    percent=False
)
cm.fit(X_train3, y_train3)
cm.score(X_test, y_test)
cm.show(outpath='cm1.png');


#%%plot
from sklearn import tree

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (5,5), dpi=1800)
tree.plot_tree(tree1,filled=True,rounded=True, ax=axes,feature_names=X.columns, class_names = ['Sem AVD','AVD'])
plt.savefig('tree.png')

