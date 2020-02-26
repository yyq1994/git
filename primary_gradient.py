import tensorflow as tf

x1 = tf.Variable(2.)
x2 = tf.Variable(3.)

# 求偏导
def g(x1,x2):
    return (x1 + 5) * (x2 ** 2)

with tf.GradientTape() as tape:         #用完一次就关闭了
    # tape.watch(x1)                    #如果x1,x2是常量的话，看起来就像是把它们当成变量来算了
    # tape.watch(x2)
    z = g(x1,x2)

dz_x1x2 = tape.gradient(z,[x1,x2])
print(dz_x1x2)            #9   42

# z1,z2
x = tf.Variable(5.)
with tf.GradientTape() as tape:
    # tape.watch(x1)            #如果x1,x2是常量的话，看起来就像是把它们当成变量来算了
    # tape.watch(x2)
    z1 = 3*x
    z2 = x**2

dz_x = tape.gradient([z1,z2],x)
print(dz_x)            #13   (3+10)


# optimizer
learning_rate = 0.1
x3 = tf.Variable(0.)
# optimizer = tf.keras.optimizers.SGD(lr=learning_rate)

for _ in range(100):
    with tf.GradientTape() as tape:
        zz = 3. *x3 **2 + 2.* x3 - 3
    dz_x3 = tape.gradient(zz,x3)
    # optimizer.apply_gradients([z,x3])
    # print(learning_rate,dz_x3)
    x3.assign_sub(learning_rate * dz_x3)               # x = x - lr*gradient    梯度下降
print(x3)

