#TRABALHO DE CONCLUSÃO DE CURSO 2
#AUTORES: ANA VITÓRIA LACHOWSKI E SABRINA BASSAI
#05/2023 - UTFPR

from numpy.core.multiarray import concatenate

import pandas as pd
import glob
import os

from scipy.signal import find_peaks, peak_prominences, peak_widths
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib.pyplot as plt


from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from sklearn import datasets
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from google.colab import drive
import os

# monta o Google Drive
drive.mount('/content/drive')

# define a pasta que você deseja abrir
pasta_NaoQueda= '/content/drive/MyDrive/TCC 2/dados_edge_nao_queda'
pasta_Queda='/content/drive/MyDrive/TCC 2/dados_edge_queda'
files_NaoQueda=os.listdir(pasta_NaoQueda)
files_Queda=os.listdir(pasta_Queda)
df_arr = None
arquivos_lidos = []
vetor_X=np.zeros(0)
vetor_Y=np.zeros(0)



def graficos(diretorio):
  for arquivo in os.listdir(diretorio):
  
    if arquivo.endswith(".csv"):
    
        # Ler o arquivo CSV em um DataFrame do pandas
        caminho_arquivo = os.path.join(diretorio, arquivo)
        df = pd.read_csv(caminho_arquivo)

        df.columns = ['Coluna 1', 'Coluna 2', 'Coluna 3', 'Coluna 4', 'Coluna 5', 'Coluna 6', 'Coluna 7']
        plt.plot(df['Coluna 1'], df['Coluna 5'], label='Eixo X')
        plt.plot(df['Coluna 1'], df['Coluna 6'], label='Eixo Y')
        plt.plot(df['Coluna 1'], df['Coluna 7'], label='Eixo Z')

        plt.title("Gráfico de " + arquivo)
        plt.xlabel("Tempo (s)")
        plt.ylabel("Ângulo em graus (°)")
        plt.show() 


graficos(pasta_NaoQueda) 
pr_value=100

for file in files_NaoQueda:
  if file.endswith(".csv"):
    filepath=os.path.join(pasta_NaoQueda,file)
    df = pd.read_csv(filepath,index_col=None,header=None)
    acc_values = df.iloc[:, 1:4].to_numpy()
    modulos = np.linalg.norm(acc_values, axis=1)
    df['Mod_acc'] = modulos
    gyr_values = df.iloc[:, 5:7].to_numpy()
    modulos_gyr = np.linalg.norm(gyr_values, axis=1)
    df['Mod_gyr'] = modulos_gyr
    
    features = np.array([0]) #Inicializa a matriz com uma linha e uma coluna com valor 0
    numberDataFrame=df.to_numpy()
    timeInterval=np.arange(0, max(numberDataFrame[:,0]), 10) #cria vetor que vai de 0 até o numero máximo da linha de tempo, com intervalos de 10 em 10
      #torna a amostra igualmente espaçada
    vetorNorma = timeInterval
    for i in range(1,5):
       vetorNorma = np.vstack((vetorNorma, np.interp(timeInterval,numberDataFrame[:,0], numberDataFrame[:,i])))
        #encontra valores de x,y,z para osvalores de tempo ajustados
        #print(vetorNorma)
    x = vetorNorma[1,:] #pega todos dos valores do eixo x

    peaks, _ = find_peaks(x, prominence=(pr_value)) #encontra os picos 
    features=np.concatenate((features, np.array([len(peaks)]))) #Armazena em um vetor o valor de picos encontrados em uma amostra

    picos=4*[0]
    picos_Altos=np.argsort(x[peaks]) #armazena valores de pico mais altos
   # print(df)
    for k in range(0, min(4,len(peaks))):
          picos[k]=x[peaks[picos_Altos[-(k+1)]]]

    features = np.concatenate((features,picos)) #Concatena picos e numero de picos
    desvio_padrao = np.std(x)
    features = np.append(features, desvio_padrao)

    if peaks.size > 0:
      max_peak_idx = np.argmax(x[peaks])
      min_peak_idx = np.argmin(x[peaks])
      dist = abs(x[peaks[max_peak_idx]] - x[peaks[min_peak_idx]])
    else:
      dist=0
    features = np.append(features, dist)


    if df_arr is None:
      df_arr = features
    else:
      df_arr = np.vstack((df_arr, features))    
    
  ##################################################################### NÃO QUEDA ###################################################################################

for file in files_Queda:
  if file.endswith(".csv"):
    filepath=os.path.join(pasta_Queda,file)
    df = pd.read_csv(filepath,index_col=None,header=None)
    features = np.array([1])
    acc_values = df.iloc[:, 1:4].to_numpy()
    modulos = np.linalg.norm(acc_values, axis=1)
    df['Mod_acc'] = modulos
    gyr_values = df.iloc[:, 5:7].to_numpy()
    modulos_gyr = np.linalg.norm(gyr_values, axis=1)
    df['Mod_gyr'] = modulos_gyr
    numberDataFrame=df.to_numpy()
    timeInterval=np.arange(0, max(numberDataFrame[:,0]), 10)
    vetorNorma = timeInterval
    #print(df)
    for i in range(1,5):
      vetorNorma = np.vstack((vetorNorma, np.interp(timeInterval,numberDataFrame[:,0], numberDataFrame[:,i])))

    x = vetorNorma[1,:]
    peaks, _ = find_peaks(x, prominence=(pr_value)) #Essa linha analisa os picos dos gráficos considerando que quando ocorre uma não-queda, o pico não tem valor superior a 0,7

    features=np.concatenate((features, np.array([len(peaks)])))
    picos=4*[0] 
    picos_Altos=np.argsort(x[peaks]) 

  
    for k in range(0, min(4,len(peaks))):
      picos[k]=x[peaks[picos_Altos[-(k+1)]]] 

    features = np.concatenate((features,picos))
    

    desvio_padrao = np.std(x)
    features = np.append(features, desvio_padrao)

    if peaks.size > 0:
      max_peak_idx = np.argmax(x[peaks])
      min_peak_idx = np.argmin(x[peaks])
      dist = abs(x[peaks[max_peak_idx]] - x[peaks[min_peak_idx]])
    else:
      dist=0
    features = np.append(features, dist)

    if df_arr is None:
      df_arr = features
    else:
       df_arr = np.vstack((df_arr, features))  



features = pd.DataFrame(df_arr, columns=['Caiu', 'numero_picos', 'AmplitudePico1', 'AmplitudePico2', 'AmplitudePico3', 'AmplitudePico4', 'Desvio_Padrao', 'Media_Dist_Picos'])
X=features[['numero_picos', 'AmplitudePico1', 'AmplitudePico2', 'AmplitudePico3', 'AmplitudePico4', 'Desvio_Padrao', 'Media_Dist_Picos']]
y=features['Caiu']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


c=1
while c<=50:
  print("C:" + str(c))
  lr = LogisticRegression(C=c, random_state=1)
  lr.fit(X_train, y_train)
  y_pred = lr.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  print('***************************************RESULTADO************************************************')
  print('Acuracia para LR com criterio LINEAR : %f \n' % accuracy_score(y_test, y_pred))
  print('Matriz de confusao: ')
  print(cm)
  recall = recall_score(y_test, y_pred)
  print('RECALL: ' + str(recall))
  F1 = f1_score(y_test, y_pred)
  print('F1 SCORE: ' + str(F1))

  svm = SVC(kernel='linear', C=c, random_state=1)
  svm.fit(X_train, y_train)
  y_pred = svm.predict(X_test)  
  cm = confusion_matrix(y_test, y_pred)
  print('***************************************RESULTADO************************************************')
  print('Acuracia para SVC com criterio LINEAR: %f \n' % accuracy_score(y_test, y_pred))
  print('Matriz de confusao: ')
  print(cm)
  recall = recall_score(y_test, y_pred)
  print('RECALL: ' + str(recall))
  F1 = f1_score(y_test, y_pred)
  print('F1 SCORE: ' + str(F1))


  tree = DecisionTreeClassifier(criterion='gini',  max_depth=2000, 
                          random_state=1)
  tree.fit(X_train, y_train)
  y_pred = tree.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  print('***************************************RESULTADO************************************************')
  print('Acuracia para Decision Tree com criterio entropy: %f \n' % accuracy_score(y_test, y_pred))
  print('Matriz de confusao: ')
  print(cm)
  recall = recall_score(y_test, y_pred)
  print('RECALL: ' + str(recall))
  F1 = f1_score(y_test, y_pred)
  print('F1 SCORE: ' + str(F1))

  svm = SVC(kernel='sigmoid', random_state=1, gamma=0.1, C=c)
  svm.fit(X_train, y_train)
  y_pred = svm.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  print('***************************************RESULTADO************************************************')
  print('Acuracia para SVC com criterio sigmoid: %f \n' % accuracy_score(y_test, y_pred))
  print('Matriz de confusao: ')
  print(cm)
  recall = recall_score(y_test, y_pred)
  print('RECALL: ' + str(recall))
  F1 = f1_score(y_test, y_pred)
  print('F1 SCORE: ' + str(F1))

  

  svm = SVC(kernel='poly', random_state=1, gamma=0.1, C=5)
  svm.fit(X_train, y_train)
  y_pred = svm.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  print('***************************************RESULTADO************************************************')
  print('Acuracia para SVC com criterio  poly: %f \n' % accuracy_score(y_test, y_pred))
  print('Matriz de confusao: ')
  print(cm)
  recall = recall_score(y_test, y_pred)
  print('RECALL: ' + str(recall))
  F1 = f1_score(y_test, y_pred)
  print('F1 SCORE: ' + str(F1))

  c=c+1

