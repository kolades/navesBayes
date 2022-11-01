import streamlit as st

from sklearn.ensemble import RandomForestClassifier 
import pandas as pd
dados = pd.read_csv('Iris.csv')

classes = dados['Species']
nomesColunas = dados.columns.to_list()
tamanho = len(nomesColunas)
nomesColunas = nomesColunas[1:tamanho-1]
features = dados[nomesColunas]

from sklearn.model_selection import train_test_split

features_treino,features_teste,classes_treino,classes_teste = train_test_split(features,
                                                                               classes,
                                                                               test_size=0.26,
                                                                               random_state=3)

floresta = RandomForestClassifier(n_estimators=90) 

floresta.fit(features_treino,classes_treino)
predicoes = floresta.predict(features_teste)


st.title('Aplicativo de IA')
SepalLengthCm = st.number_input('Digite o comprimento do caule')
SepalWidthCm = st.number_input('Digite a largura do caule')
PetalLengthCm = st.number_input('Digite o comprimento da petala')
PetalWidthCm = st.number_input('Digite a largura da petala')
if st.button('Clique aqui'):
  resultado = floresta.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
  st.write('Resultado:',resultado)
