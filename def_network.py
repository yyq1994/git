import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

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

class CustomizeDenseLayer(keras.layers.Layer):
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

customized_softplus = keras.layers.Lambda(lambda x:tf.nn.softplus(x))

model = keras.models.Sequential()
model.add(CustomizeDenseLayer(30,'relu',input_shape=x_train_scaled.shape[1:]))
customized_softplus
model.add(CustomizeDenseLayer(30,'relu'))
customized_softplus
model.add(CustomizeDenseLayer(30,'relu'))
customized_softplus
model.add(CustomizeDenseLayer(1))
# 模型编译
model.compile(loss='mean_squared_error',optimizer='adam')      #optimizer我用了 'sgd'几个epochloss就变成了nan了
callbacks = [keras.callbacks.EarlyStopping(patience=5,min_delta=1e-2)]

#模型训练
history = model.fit(x_train_scaled,y_train,validation_data=(x_valid_scaled,y_valid),epochs=100,callbacks=callbacks)

# 效果绘制
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()

plot_learning_curves(history)

 # 评估
print(model.evaluate(x_test_scaled,y_test))