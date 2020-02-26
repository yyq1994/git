import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
#将python的函数编译成tensorflow的图结构
@tf.function(input_signature=[tf.TensorSpec([None],tf.int32,name='x')])         #限定函数输入类型
def cube(z):
    return tf.pow(z,3)

# print(cube(tf.constant([1,2,3])))

 #保存图结构
cube_int32 = cube.get_concrete_function(tf.TensorSpec([None],tf.int32,name='x'))
print(cube_int32)

print(cube_int32.graph)      #方法的图


print('第二个结构：',cube_int32.graph.get_operations())
exit()
op = cube_int32.graph.get_operations()[2]           #pow
print('输入：',list(op.inputs))
print('输出：',list(op.outputs))

print(cube_int32.graph.get_operation_by_name("x"))      #通过名字获取操作
print(cube_int32.graph.get_tensor_by_name("x:0"))       #通过名字获取张量
print(cube_int32.graph.as_graph_def())                  #查看节点