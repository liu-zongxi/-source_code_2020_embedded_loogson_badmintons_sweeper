import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])#dense定义了一层相连的神经元，一层，一个单元 ，sequential顺序
model.compile(optimizer='sgd',loss='mean_squared_error')#优化器，损失函数，sgd只是一种，还有很多优化器，同样损失函数

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)#dtype使数据类型设为float
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

model.fit(xs, ys, epochs=500)#fit命令中训练，训练500次
print(model.predict([10.0]))