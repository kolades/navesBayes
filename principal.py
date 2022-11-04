import streamlit as st
import pandas as pd #manipulacao de dados

###################importando os dados do csv ########################

dados = pd.read_csv('Iris.csv')

##############separar as classes das features  #######################

classes = dados['Species']
nomesColunas = dados.columns.to_list()
tamanho = len(nomesColunas)#quantos nomes tem
nomesColunas = nomesColunas[:tamanho-1]#retira o ultimo
features = dados[nomesColunas]#monta o features
features.pop('Id') 

#######[2.1] Formação dos conjuntos de treinamento e teste ##########
import numpy as np
from sklearn.model_selection import train_test_split

data = np.array(dados)

data = data[:, 1:] # remover essa linha caso a primeira coluna corresponda a dados relevantes

# Indentificando os rótulos das classes
labels = []
for line in range(data.shape[0]):
  if(labels.count(data[line, data.shape[1]-1])==0):
    labels.append(data[line, data.shape[1]-1])

y = np.array(data[:, data.shape[1]-1])
x = (data[:, :(data.shape[1]-1)]).astype(np.float32)

# Gerando os conjuntos de treinamento e teste (validação)
train_x, test_x, train_y, test_y = train_test_split(x,y, test_size=0.25) # 0.25 dos dados no conjunto de teste

########### [3] Ajuste do modelo para classes com distribuição gaussiana #################
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()


model.fit(train_x, train_y)

st.title('Aplicativo Naves de IA')
SepalLengthCm = st.number_input('Digite o comprimento do caule')
SepalWidthCm = st.number_input('Digite a largura do caule')
PetalLengthCm = st.number_input('Digite o comprimento da petala')
PetalWidthCm = st.number_input('Digite a largura da petala')
if st.button('Clique aqui'):
  resultado = model.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
  st.write('Resultado:',resultado)
 
  if resultado == "Iris-versicolor":
   st.write("Iris-versicolor")
   st.image("Iris-versicolor.jpg")
  
  if resultado == "Iris-setosa":
   st.write("Iris-setosa")
   st.image("Iris-setosa.jpg")
  
 if resultado == "Iris-virginica":
  st.write("Iris-virginica")
  st.image("Iris-virginica.jpg")
  
  
