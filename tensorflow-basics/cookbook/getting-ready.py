'''
Created on Mar 4, 2018

@author: Dr.Guo
'''
import tensorflow as tf
from cookbook.data_declare import sess
from test.test_threading_local import target
sess=tf.Session()

import numpy as np

x_vals=np.array([1.,3.,5.,7.,9.])
x_data=tf.placeholder(tf.float32)
m_const=tf.constant(3.)
m_product=tf.mul(x_data,m_const)
for x_val in x_vals:
    print(sess.run(m_product,feed_dict={x_data:x_val}))

'''
Layering Nested Operations
'''
    
my_array=np.array([[],
                  [],
                  []])
x_vals=np.array([my_array,my_array+1])
x_data=tf.placeholder(tf.float32,shape=(3,5))
'''
x_data=tf.placeholder(tf.float32,shape=(3,None))
'''

m1=tf.constant([[],[],[],[],[]])
m2=tf.constant([[]])
a1=tf.constant([[]])

prod1=tf.matmul(x_data,m1)
prod2=tf.matmul(prod1,m2)
add1=tf.add(prod2,a1)

for x_val in x_vals:
    print(sess.run(add1,feed_dict={x_data:x_val}))



'''
Working with multiple layers
import tensorflow as tf
import numpy as np
sess=tf.Session()
'''

x_shape=[1,4,4,1]
x_val=np.random.uniform(size=x_shape)

x_data=tf.placeholder(tf.float32,shape=x_shape)
my_filter=tf.constant(0.25,shape=[2,2,1,1])
my_strides=[1,2,2,1]
mov_avg_layer=tf.nn.conv2d(x_data,my_filter,my_strides,padding='SAME''',name="Moving_Avg_Window")

def custom_layer(input_matrix):
    input_matrix_sqeezed=tf.squeeze(input_matrix)
    A=tf.constant([[1.,2.],[-1.,3.]])
    b=tf.constant(1.,shape=[2,2])
    temp1=tf.matmul(A,input_matrix_sqeezed)
    temp=tf.add(temp1,b)
    return (tf.sigmoid(temp))

with tf.name_scope("Custom_Layer") as scope:
    custom_layer1=custom_layer(mov_avg_layer)
    print(sess.run(custom_layer1,feed_dict={x_data:x_val}))



'''
Implementing loss functions
import tensorflow as tf
import numpy as np
sess=tf.Session()
'''

import matplotlib.pyplot as plt
import tensorflow as tf

x_vals=tf.linspace(-1.,1.,500)
target- tf.constant(0.)

l2_y_vals=tf.square(target-x_vals)
l2_y_out=sess.run(l2_y_vals)

l1_y_vals=tf.abs(target-x_vals)
l1_y_out=sess.run(l1_y_vals)

delta1=0.25
delta2=5

delta1=tf.constant(0.25)

phuber1_y_vals=tf.mul(tf.square(delta1),tf.sqrt(1.+tf.square((target-x_vals)/delta1))-1., weaklist))))))

phuber1_y_out=sess.run(phuber1_y_vals))

delta2=tf.constant(5)

phuber2_y_vals=tf.mul(tf.square(delta1),tf.sqrt(1.+tf.square((target-x_vals)/delta1))-1., weaklist))))))

phuber2_y_out=sess.run(phuber2_y_vals))

x_vals=tf.linspace(-3.,5,500)
target=tf.constant(1.))
targets=tf.fill([500,],1.)

'''
Hinge loss is mostly used for SVMs, but can be used in neural networks as well.

'''

hinge_y_vals=tf.maximum(0., 1.-tf.mul(target,x_vals))
hinge_y_out=sess.run(hinge_y_vals)


'''
Cross entropy loss for a binary case is also sometimes referred to as the logistic loss function.
'''
xentropy_y_vals=-tf.mul(target,tf.log(x_vals))-tf.mul(1.-target),tf.log(1.-x_vals))

xentropy_y_out=sess.run(xentropy_y_vals)


'''
8.Sigmoid cross entropy loss is very similar to the previous loss function except we transform the x-values by the sigmoid funciton before we put them in the cross entropy loss.

'''
xentropy_sigmoid_y_vals=tf.nn.sigmoid_cross_entropy_with_logits(x_vals, , targets)
xentropy_sigmoid_y_out=sess.run(xentropy_sigmoid_y_vals)

'''
9.Weighted loss
'''



'''
10.Softmax cross-entropy loss
'''


'''
11.Sparse Softmax cross-entropy loss
'''
x_array=sess.run(x_vals)
plt.plot(x_array,12_y_out,'b-',label='L2 Loss')























