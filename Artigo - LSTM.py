import pandas as pd
import matplotlib as plt

base = pd.read_excel('juramento.xls', usecols=[2])


'''
#PREPROCESSAMENTO
#base2 = base['COTA N.A.'].astype(float)


base.drop([      'MÊS',  # Retirando as colunas não utilizadas
                 'VOLUME M3', 
                 '% VOLUME DISPONÍVEL', 
                 'PRECIPITAÇÃO (mm)',
                 'VAZÃO MÉDIA CAPTADADA (l/s)'], inplace=True, axis=1)

'''
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range = (0,1)) # Normalizando os dados

base_normalizada = normalizador.fit_transform(base)

import matplotlib.pyplot as plt
plt.plot(base)

periodos = 30
previsão_futura = 1 #horizonte das previsãoes

x = base_normalizada[0:(len(base_normalizada)- (len(base_normalizada) % periodos))]
x_batches = x.reshape(-1, periodos, 1) 

y = base_normalizada[1:(len(base_normalizada) - (len(base_normalizada) % periodos)) + previsão_futura]
y_batches = y.reshape(-1, periodos, 1)

x_teste = base_normalizada[-(periodos + previsão_futura):]
x_teste = x_teste[:periodos]
x_teste = x_teste.reshape(-1, periodos, 1)
y_teste = base_normalizada[-(periodos):]
y_teste = y_teste.reshape(-1, periodos, 1)

#CONSTRUINDO A REDE

import tensorflow as tf
tf.reset_default_graph()

entradas = 1
neuronios_oculta = 128
neuronios_saida = 1

xph = tf.placeholder(tf.float32, [None, periodos, entradas])
yph = tf.placeholder(tf.float32, [None, periodos, neuronios_saida])

celula = tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)
# camada saída
celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)

def cria_uma_celula():
    return tf.contrib.rnn.LSTMCell(num_units = neuronios_oculta, activation = tf.nn.relu)

def cria_varias_celulas():
    return  tf.nn.rnn_cell.MultiRNNCell([cria_uma_celula() for i in range(2)]) 

#def cria_varias_celulas():
#    celulas =  tf.nn.rnn_cell.MultiRNNCell([cria_uma_celula() for i in range(4)])
#    return tf.contrib.rnn.DropoutWrapper(celulas, output_keep_prob = 0.01)

celula = cria_varias_celulas()
# camada saída
celula = tf.contrib.rnn.OutputProjectionWrapper(celula, output_size = 1)


saida_rnn, _ = tf.nn.dynamic_rnn(celula, xph, dtype = tf.float32)
erro = tf.losses.mean_squared_error(labels = yph, predictions = saida_rnn)
otimizador = tf.train.AdamOptimizer(learning_rate = 0.001)
treinamento = otimizador.minimize(erro)
erro2 = []

with tf.Session() as sess:
    import time
    start_time = time.time()
    

    sess.run(tf.global_variables_initializer())
    
    for epoca in range(20000):
        _, custo = sess.run([treinamento, erro], feed_dict = {xph: x_batches, yph: y_batches})
        erro2.append(custo)
        if epoca % 100 == 0:
            print(epoca + 1, ' erro: ', custo)
    
    previsoes = sess.run(saida_rnn, feed_dict = {xph: x_teste})
tempo = (time.time() - start_time)
tempo = round(tempo, 2)

import numpy as np
y_teste.shape
y_teste2 = np.ravel(y_teste)

previsoes2 = np.ravel(previsoes)

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_teste2, previsoes2)

normalizador.inverse_transform(previsoes2)
normalizador.inverse_transform(y_teste2)


plt.plot(y_teste2,  label = 'Valor real')
plt.plot(previsoes2,  label = 'Previsões')
plt.xlabel('Período')
plt.ylabel('nível')
plt.legend()

plt.plot(erro2, label = 'custo')
plt.legend()