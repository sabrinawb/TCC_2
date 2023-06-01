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
from sklearn.metrics import f1_score, recall_score
from google.colab import drive
import os

# monta o Google Drive
drive.mount('/content/drive')

# define a pasta que você deseja abrir
pasta_NaoQueda= '/content/drive/MyDrive/TCC 2/nao_queda_peito'
pasta_Queda='/content/drive/MyDrive/TCC 2/queda_peito' 
files_NaoQueda=os.listdir(pasta_NaoQueda)
files_Queda=os.listdir(pasta_Queda)
df_arr = None
arquivos_lidos = []
vetor_X=np.zeros(0)
vetor_Y=np.zeros(0)
step=0.08

def graficos(diretorio):
  for arquivo in os.listdir(diretorio):
  
    if arquivo.endswith(".csv"):
    
        # Ler o arquivo CSV em um DataFrame do pandas
        caminho_arquivo = os.path.join(diretorio, arquivo)

        df = pd.read_csv(caminho_arquivo)

        dados = df.values
        # Criar o gráfico
       # plt.plot(df['Coluna 1'], df['Coluna 2'], label='Eixo X')
     #   plt.plot(df['Coluna 1'], df['Coluna 3'], label='Eixo Y')
      #  plt.plot(df['Coluna 1'], df['Coluna 4'], label='Eixo Z')
        eixo_x = dados[:, 0]
        eixo_y = dados[:, 1:4]

            # Criar o gráfico
        plt.plot(eixo_x, eixo_y[:, 0], label='Eixo X')
        plt.plot(eixo_x, eixo_y[:, 1], label='Eixo Y')
        plt.plot(eixo_x, eixo_y[:, 2], label='Eixo Z')
        plt.title("Gráfico de " + arquivo)
        plt.xlabel("Eixo X")
        plt.ylabel("Eixo Y")
        plt.show() 


pr_value_gyr=4
pr_value_acc=1

for file in files_NaoQueda:
  if file.endswith(".csv"):
    filepath=os.path.join(pasta_NaoQueda,file)
    df = pd.read_csv(filepath,index_col=None,header=None)
    features = np.array([0]) 
    numberDataFrame=df.to_numpy()
    timeInterval=np.arange(0, max(numberDataFrame[:,0]), 0.08) 
    
  
    vetorNorma1 = timeInterval
    vetorNorma2 = timeInterval

    for i in range(1,3):
       vetorNorma1 = np.vstack((vetorNorma1, np.interp(timeInterval,numberDataFrame[:,0], numberDataFrame[:,i])))
    for i in range(4,6):
       vetorNorma2 = np.vstack((vetorNorma2, np.interp(timeInterval,numberDataFrame[:,0], numberDataFrame[:,i])))

    Accx = vetorNorma1[1,:]
    Gyrx= vetorNorma2[1,:] 


    peaksAccx, _ = find_peaks(Accx,prominence=pr_value_acc)
    results_half_Accx = peak_widths(Accx, peaksAccx, rel_height=0.5)
    

    picosAccx=3*[0]
    l_Accx = 3*[0]
    picos_AltosAccx=np.argsort(Accx[peaksAccx]) 
    for k in range(0, min(3,len(peaksAccx))):
          
          l_Accx[k]=results_half_Accx[0][picos_AltosAccx[-(k+1)]]
  
    features = np.concatenate((features,l_Accx)) 

    dx_acc = np.diff(Accx)/step 
    pk_pk_acc = [0]*(len(dx_acc)-14)
    for k in range(0,len(dx_acc)-14):
        sig = dx_acc[k:k+14]
        pk_pk_acc[k] = np.abs(np.max(sig)-np.min(sig))                             
    
    max_pk_acc = np.unique(pk_pk_acc)[::-1][:4]    
    features = np.concatenate((features,max_pk_acc)) 


    if peaksAccx.size > 0:
      max_peak_idx = np.argmax(Accx)
      min_peak_idx = np.argmin(Accx)
      dist = abs(Accx[max_peak_idx] - Accx[min_peak_idx])
    else:
      dist=0
    features = np.append(features, dist)
    std_acc=np.std(abs(Accx))
    features = np.append(features, std_acc)
#################################################### GIROSCOPIO ###################################################################################


    Gyrx= vetorNorma2[1,:] 
    peaksGyr, _ = find_peaks(Gyrx, prominence=pr_value_gyr) #encontra os picos 
    results_half_Gyrx = peak_widths(Gyrx, peaksGyr, rel_height=0.5)
    

    picosGyrx=3*[0]
    l_Gyrx=3*[0]
    picos_AltosGyrx=np.argsort(Gyrx[peaksGyr]) #armazena valores de pico mais altos
    for k in range(0, min(3,len(peaksGyr))):
          picosGyrx[k]=Gyrx[peaksGyr[picos_AltosGyrx[-(k+1)]]]
          l_Gyrx[k]=results_half_Gyrx[0][picos_AltosGyrx[-(k+1)]]
    features = np.concatenate((features,picosGyrx)) #Concatena picos e numero de picos
    features = np.concatenate((features,l_Gyrx)) 

    dx_gyr = np.diff(Gyrx)/step 
    pk_pk_gyr = [0]*(len(dx_gyr)-14)
    for k in range(0,len(dx_gyr)-14):
        sig_gyr = dx_gyr[k:k+14]
        pk_pk_gyr[k] = np.abs(np.max(sig_gyr)-np.min(sig_gyr))                             
    
    max_pk_gyr = np.unique(pk_pk_gyr)[::-1][:4]    
    features = np.concatenate((features,max_pk_gyr))
    
    if peaksGyr.size > 0:
      max_peak_idx = np.argmax(Gyrx)
      min_peak_idx = np.argmin(Gyrx)
      dist = abs(Gyrx[max_peak_idx] - Gyrx[min_peak_idx])
    else:
      dist=0
    features = np.append(features, dist)

    std_gyr=np.std(abs(Gyrx))
    #features = np.append(features, std_gyr)

    if df_arr is None:
      df_arr = features
    else:
      df_arr = np.vstack((df_arr, features))    

####################################################################################QUEDA############################################################################

for file in files_Queda:
  if file.endswith(".csv"):
    filepath=os.path.join(pasta_Queda,file)
    df = pd.read_csv(filepath,index_col=None,header=None)
    features = np.array([1]) #Inicializa a matriz com uma linha e uma coluna com valor 0
    numberDataFrame=df.to_numpy()

    timeInterval=np.arange(0, max(numberDataFrame[:,0]), 0.08) #cria vetor que vai de 0 até o numero máximo da linha de tempo, com intervalos de 10 em 10
      #torna a amostra igualmente espaçada
    vetorNorma1 = timeInterval
    vetorNorma2 = timeInterval

    for i in range(1,3):
       vetorNorma1 = np.vstack((vetorNorma1, np.interp(timeInterval,numberDataFrame[:,0], numberDataFrame[:,i])))
    for i in range(4,6):
       vetorNorma2 = np.vstack((vetorNorma2, np.interp(timeInterval,numberDataFrame[:,0], numberDataFrame[:,i])))

    Accx = vetorNorma1[1,:] #pega todos dos valores do eixo x
    Gyrx= vetorNorma2[1,:] #pega todos dos valores do eixo x

    peaksAccx, _ = find_peaks(Accx,prominence=pr_value_acc) #encontra os picos
    results_half_Accx = peak_widths(Accx, peaksAccx, rel_height=0.5)

    picosAccx=3*[0]
    l_Accx = 3*[0]
    picos_AltosAccx=np.argsort(Accx[peaksAccx]) #armazena valores de pico mais altos
    for k in range(0, min(3,len(peaksAccx))):
         # picosAccx[k]=Accx[peaksAccx[picos_AltosAccx[-(k+1)]]]
          l_Accx[k]=results_half_Accx[0][picos_AltosAccx[-(k+1)]]
  
    features = np.concatenate((features,l_Accx)) 

    dx_acc = np.diff(Accx)/step 
    pk_pk_acc = [0]*(len(dx_acc)-14)
    for k in range(0,len(dx_acc)-14):
        sig = dx_acc[k:k+14]
        pk_pk_acc[k] = np.abs(np.max(sig)-np.min(sig))                             
    
    max_pk_acc = np.unique(pk_pk_acc)[::-1][:4]    
    features = np.concatenate((features,max_pk_acc)) 
    
    
    if peaksAccx.size > 0:
      max_peak_idx = np.argmax(Accx)
      min_peak_idx = np.argmin(Accx)
      dist = abs(Accx[max_peak_idx] - Accx[min_peak_idx])
    else:
      dist=0
    features = np.append(features, dist)

    std_acc=np.std(abs(Accx))
    features = np.append(features, std_acc)
###########################################################################GIROSCOPIO########################################################3
    
    
    Gyrx= vetorNorma2[1,:] #pega todos dos valores do eixo x
    peaksGyr, _ = find_peaks(Gyrx,prominence=pr_value_gyr) #encontra os picos 
    results_half_Gyrx = peak_widths(Gyrx, peaksGyr, rel_height=0.5)
   

    picosGyrx=3*[0]
    l_Gyrx=3*[0]
    picos_AltosGyrx=np.argsort(Gyrx[peaksGyr]) #armazena valores de pico mais altos
    for k in range(0, min(3,len(peaksGyr))):
          picosGyrx[k]=Gyrx[peaksGyr[picos_AltosGyrx[-(k+1)]]]
          l_Gyrx[k]=results_half_Gyrx[0][picos_AltosGyrx[-(k+1)]]
    features = np.concatenate((features,picosGyrx)) #Concatena picos e numero de picos
    features = np.concatenate((features,l_Gyrx)) 

    dx_gyr = np.diff(Gyrx)/step 
    pk_pk_gyr = [0]*(len(dx_gyr)-14)
    for k in range(0,len(dx_gyr)-14):
        sig_gyr = dx_gyr[k:k+14]
        pk_pk_gyr[k] = np.abs(np.max(sig_gyr)-np.min(sig_gyr))                             
    
    max_pk_gyr = np.unique(pk_pk_gyr)[::-1][:4]    
    features = np.concatenate((features,max_pk_gyr)) 
    
    if peaksGyr.size > 0:
      max_peak_idx = np.argmax(Gyrx)
      min_peak_idx = np.argmin(Gyrx)
      dist = abs(Gyrx[max_peak_idx] - Gyrx[min_peak_idx])
    else:
      dist=0
    features = np.append(features, dist)

    std_gyr=np.std(abs(Gyrx))

    if df_arr is None:
      df_arr = features
    else:
      df_arr = np.vstack((df_arr, features))    
    

features = pd.DataFrame(df_arr, columns=['Caiu','LarguraAccx1','LarguraAccx2', 'LarguraAccx3', 'DerivadaAcc1', 'DerivadaAcc2', 'DerivadaAcc3', 'DerivadaAcc4','Media_Dist_Picos_Accx','DesvioPadrao_Acc', 'AmplitudePicoGyr1', 'AmplitudePicoGyr2', 'AmplitudePicoGyr3', 'LarguraGyr1','LarguraGyr2', 'LarguraGyr3', 'DerivadaGyr1', 'DerivadaGyr2', 'DerivadaGyr3', 'DerivadaGyr4','Media_Dist_Picos_Gyr'])
X=features[['LarguraAccx1','LarguraAccx2', 'LarguraAccx3', 'DerivadaAcc1', 'DerivadaAcc2', 'DerivadaAcc3', 'DerivadaAcc4','Media_Dist_Picos_Accx','AmplitudePicoGyr1', 'AmplitudePicoGyr2', 'AmplitudePicoGyr3', 'LarguraGyr1','LarguraGyr2', 'LarguraGyr3', 'DerivadaGyr1', 'DerivadaGyr2', 'DerivadaGyr3', 'DerivadaGyr4','Media_Dist_Picos_Gyr']]
y=features['Caiu']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)
print(features)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)



c=50
while c<=50:
  print("C:" + str(c))
  lr = LogisticRegression(C=c, random_state=0)
  lr.fit(X_train_std, y_train)
  y_pred = lr.predict(X_test_std)
  cm = confusion_matrix(y_test, y_pred)
  #print("Para um PR_VALUE=" +str(pr_value))

  print("********************************************* RESULTADOS ********************************************************")
  print('Matriz de confusao da logistc regression')
  print(cm)
  F1 = f1_score(y_test, y_pred)
  recall =  recall_score(y_test, y_pred)
  print("F1 SCORE: " + str(F1))
  print("recall: " + str(recall))
  print('Acuracia PARA LOGISTIC REGRESSION: %f' % accuracy_score(y_test, y_pred))

  svm = SVC(kernel='linear', C=c, random_state=1)
  svm.fit(X_train_std, y_train)
  y_pred = svm.predict(X_test_std)  
  cm = confusion_matrix(y_test, y_pred)
  print("********************************************* RESULTADOS ********************************************************")
  print('Matriz de confusao para SVS Linear')
  print(cm)
  F1 = f1_score(y_test, y_pred)
  recall =  recall_score(y_test, y_pred)
  print("F1 SCORE: " + str(F1))
  print("recall: " + str(recall))
  print('Acuracia PARA svc linear: %f' % accuracy_score(y_test, y_pred))

  svm = SVC(kernel='rbf', random_state=1, gamma=390, C=c)
  svm.fit(X_train_std, y_train)
  y_pred = svm.predict(X_test_std)
  cm = confusion_matrix(y_test, y_pred)
  print("********************************************* RESULTADOS ********************************************************")
  print('Matriz de confusao para SVM RBF')
  print(cm)
  F1 = f1_score(y_test, y_pred)
  recall =  recall_score(y_test, y_pred)
  print("F1 SCORE: " + str(F1))
  print("recall: " + str(recall))
  print('Acuracia PARA svm rbf: %f' % accuracy_score(y_test, y_pred))

  svm = SVC(kernel='sigmoid', random_state=1, gamma=0.8, C=c)
  svm.fit(X_train_std, y_train)
  y_pred = svm.predict(X_test_std)
  cm = confusion_matrix(y_test, y_pred)
  print("********************************************* RESULTADOS ********************************************************")
  print('Matriz de confusao para SVM Sigmoid')
  print(cm)
  F1 = f1_score(y_test, y_pred)
  recall =  recall_score(y_test, y_pred)
  print("F1 SCORE: " + str(F1))
  print("recall: " + str(recall))
  print('Acuracia PARA svm sigmoid: %f' % accuracy_score(y_test, y_pred))


  svm = SVC(kernel='poly', random_state=1, gamma=500, C=c)
  svm.fit(X_train_std, y_train)
  y_pred = svm.predict(X_test_std)
  cm = confusion_matrix(y_test, y_pred)
  print("********************************************* RESULTADOS ********************************************************")
  print('Matriz de confusao para SVM Poly')
  print(cm)
  F1 = f1_score(y_test, y_pred)
  recall =  recall_score(y_test, y_pred)
  print("F1 SCORE: " + str(F1))
  print("recall: " + str(recall))
  print('Acuracia PARA svm poly: %f' % accuracy_score(y_test, y_pred))

  tree = DecisionTreeClassifier(criterion='entropy',  max_depth=50, 
                          random_state=1)
  tree.fit(X_train_std, y_train)
  y_pred = tree.predict(X_test_std)
  cm = confusion_matrix(y_test, y_pred)
  print("********************************************* RESULTADOS ********************************************************")
  print('Matriz de confusao Decision Tree - Entropy')
  print(cm)
  F1 = f1_score(y_test, y_pred)
  recall =  recall_score(y_test, y_pred)
  print("F1 SCORE: " + str(F1))
  print("recall: " + str(recall))
  print('Acuracia PARA decision %f' % accuracy_score(y_test, y_pred))

  c=c+100


tree = DecisionTreeClassifier(criterion='gini',  max_depth=50, 
                          random_state=1)
tree.fit(X_train_std, y_train)
y_pred = tree.predict(X_test_std)
cm = confusion_matrix(y_test, y_pred)
print('Acuracia para Decision Tree com criterio gini: %f \n' % accuracy_score(y_test, y_pred))
print(cm)