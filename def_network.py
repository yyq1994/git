import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

# 下载数据----------加利福尼亚房价
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

# 分割数据
from sklearn.model_selection import train_test_split
x_train_all,x_test,y_train_all,y_test =train_test_split(housing.data,housing.target,random_state=7)
x_train,x_valid,y_train,y_valid = train_test_split(x_train_all,y_train_all,random_state=11)

# 归一化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# 定义网络层次
class CustomizeDenseLayer(keras.layers.Layer):                  #改动
    def __init__(self,unit,activation=None,**kwargs):
        super(CustomizeDenseLayer, self).__init__(**kwargs)
        self.unit = unit
        self.activation = keras.layers.Activation(activation)
    def build(self, input_shape):
        self.kernal = self.add_weight(name='kenal',
                                      shape=(input_shape[1],self.unit),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=self.unit,
                                    initializer='zero',
                                    trainable=True)
        super(CustomizeDenseLayer, self).build(input_shape)
    def call(self, x):
        return self.activation(x @ self.kernal + self.bias)

customized_softplus = keras.layers.Lambda(lambda x:tf.nn.softplus(x))                   #改动

model = keras.models.Sequential()
model.add(CustomizeDenseLayer(30,'relu',input_shape=x_train_scaled.shape[1:]))          #改动
customized_softplus                                                                     #改动
model.add(CustomizeDenseLayer(30,'relu'))                                               #改动
customized_softplus                                                                     #改动
model.add(CustomizeDenseLayer(30,'relu'))                                               #改动
customized_softplus                                                                     #改动
model.add(CustomizeDenseLayer(1))                                                       #改动

# 模型编译
# 定义参数
epochs = 10
batch_size = 32
step_per_epoch = len(x_train_scaled) // batch_size
optimizer = keras.optimizers.Adam()
metric = keras.metrics.MeanSquaredError()
# 取数据
def random_banch(x,y,batch_size = 32):
    idx = np.random.randint(0,len(x),batch_size)
    return x[idx],y[idx]

#模型训练
for epoch in range(epochs):
    metric.reset_states()
    for step in range(step_per_epoch):
        x_batch,y_batch = random_banch(x_train_scaled,y_train_all)
        with tf.GradientTape() as tape:
            y_pred = model(x_batch)
            loss = tf.reduce_mean(
                keras.losses.mean_squared_error(y_batch,y_pred)
            )

            metric(y_batch,y_pred)
            # print(loss.shape,metric.input_shape)
        grads = tape.gradient(loss,model.variables)
        grads_vars = zip(grads,model.variables)
        # 梯度更新
        optimizer.apply_gradients(grads_vars)
        print('epoch:{}    ,train_metric:{},\ttrain_loss: {}'.format(epoch,metric.result().numpy(),loss),end=' ')
        y_valid_pre = model(x_valid_scaled)
        valid_loss = tf.reduce_mean(keras.losses.mean_squared_error(
            y_valid,y_valid_pre
        ))
        print('\t valid_loss:{}'.format(valid_loss.numpy()))