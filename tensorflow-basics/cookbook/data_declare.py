'''
Created on Mar 4, 2018

@author: Dr.Guo
'''
import tensorflow as tf
import numpy as np
from PIL.ImageChops import constant
from jinja2.runtime import identity

'''
1.Fixed tensors
'''

row_dim,col_dim=10,10
zero_tsr=tf.zeros([row_dim,col_dim], float, "zero_tsr")
ones_tsr=tf.ones([row_dim,col_dim], float, "ones_tsr")
filled_tsr=tf.fill([row_dim,col_dim], float, "ones_tsr")
constant_tsr=tf.fill([1,2,3], float, "ones_tsr")

'''
2.similar tensors
'''

zero_similar=tf.zeros_like(constant_tsr)
ones_similar=tf.ones_like(constant_tsr)

'''
3.sequence tensors
'''

linear_tsr=tf.linspace(start=0,stop=1,start=3)
#The results will be [0.0,0.5,1.0]
integer_seq_tsr=tf.range(start=6,limet=5,delta=3)
#The results will be [6,9,12]
ones_similar=tf.ones_like(constant_tsr)
filled_tsr=tf.fill([row_dim,col_dim], float, "ones_tsr")
constant_tsr=tf.fill([1,2,3], float, "ones_tsr")

'''
4.random tensors
'''

row_dim,col_dim=10,10
randunif_tsr=tf.random_uniform([row_dim,col_dim],mnival=0,maxal=1)
#(minval<=x<=maxval)
randnorm_tsr=tf.random_normal([row_dim,col_dim],mean=0.0,stddev=1.0)
runcnorm_tsr=tf.truncated_normal([row_dim,col_dim],mean=0.0,stddev=1.0)

input_tensor=None
shuffled_output=tf.random_shuffle(input_tensor)

input_tensor, crop_size=None,None
croped_output=tf.random_crop(input_tensor,crop_size)

my_image,height,width=None,None,None
croped_image=tf.random_crop(my_image,[height/2,width/2])

'''
5.Variables
Once we have created tensorsm then we create the Vars by wrapping the tensor in the Variable() function as follows.
'''

my_var=tf.Variable(tf.zeros([row_dim,col_dim]))

numpy_01=np.zeros()
converted_tensor=tf.convert_to_tensor(numpy_01)

'''
6.Examples
'''

my_var=tf.Variable(tf.zeros([2,3]))
sess=tf.Session()
initialize_op=tf.global_variables_initializer()
sess.run(initialize_op)

'''
7.Placeholders
'''

my_var=tf.Variable(tf.zeros([2,3]))
sess=tf.Session()
x=tf.placeholder(tf.float32, shape=[2,3], name="x")
y=tf.identity(x)# This seems like as equal to y=x
x_vals=np.random.rand(2,2)
sess.run(y, feed_dict={x:x_vals})

'''
Note that sess.run() will result in a self-referencing error.
'''
initialize_op=tf.global_variables_initializer()
sess=tf.Session()

sess.run(initialize_op)

'''
8.Initialize
'''

sess=tf.Session()
first_var=tf.Variable(tf.zeros([2,3]))
sess.run(first_var.initializer)
second_var=tf.Variable(tf.zeros_like(first_var))
sess.run(second_var.initializer)

'''
9.Matrix Operations/Create Matrix
import tensorflow as tf
sess=tf.Session()
'''

identity_matrix=tf.diag([1.0,1.0,1.0])
A=tf.truncated_normal([2,3])
B=tf.fill([2,3],5.0)
C=tf.random_uniform([3,2])
D=tf.convert_to_tensor(np.array([[1.,2.,3.],[-3.,-7.,-1.],[0.,5.,-2.]]))
print(sess.run(identity_matrix))
'''
[
[1.,0.,0.]
[0.,1.,0.]
[0.,0.,1.]
]
'''
print(sess.run(A))
'''
[
[1.,0.,0.]
[0.,1.,0.]
[0.,0.,1.]
]
'''
print(sess.run(B))
'''
[
[5.,5.,5.]
[5.,5.,5.]
]
'''
print(sess.run(C))
print(sess.run(D))

'''
9.Matrix Operations
'''
print(sess.run(A+B))
print(sess.run(B-B))
print(sess.run(tf.matmul(B,identity_matrix)))
print(sess.run(tf.transpose(C)))
print(sess.run(tf.matrix_determinant(D)))
print(sess.run(tf.matrix_inverse(D)))

print(sess.run(tf.cholesky(identity_matrix)))
print(sess.run(tf.self_adjoint_eig(D)))

'''
10.Other Operations
'''
print(sess.run(tf.div(3,4)))
print(sess.run(tf.truediv(3,4)))
print(sess.run(tf.floordiv(3,4)))
print(sess.run(tf.mod(22.0,5.0)))

'''
The cross product between two tensors is achieved by the cross()
'''
print(sess.run(tf.cross([1.,0.,0.],[0.,1.,0.])))


print(sess.run(B-B))
print(sess.run(tf.matmul(B,identity_matrix)))
print(sess.run(tf.transpose(C)))
print(sess.run(tf.matrix_determinant(D)))
print(sess.run(tf.matrix_inverse(D)))

print(sess.run(tf.cholesky(identity_matrix)))
print(sess.run(tf.self_adjoint_eig(D)))

'''
commmon math functions
'''
tf.abs()
tf.ceil()
tf.cos()
tf.exp()
tf.floor()
tf.inv()
tf.log()
tf.maximum()
tf.minimum()
#tf.neg()
tf.pow()
tf.round()
tf.rsqrt()
tf.sign()
tf.sin()
tf.sqrt()
tf.square()

'''
special math functions
'''
tf.digamma()
tf.erf()
tf.erfc()
tf.igamma()
tf.igammac()
tf.lbeta()
tf.lgamma()
tf.squared_difference()

'''
Tangent function (tan(pi/4)=1)
'''
print(sess.run(tf.div(tf.div(tf.sin(3.1416/4.),tf.cos(3.1416/4.)))))


def custom_polynomial(value):
    return (tf.sub(3*tf.square(value),value)+10)
print(sess.run(custom_polynomial(11)))


'''
11.Activation Functions
---------------for introducing nonliearities in neural networks or other computational graphs in the future.
'''
import tensorflow as tf
sess=tf.Session()
'''
predefined activation functions
(1) Rectified linear unit, known as ReLU, is the most commmon and basic way to introduce a non-linear into neural networks.
The function just max(0,x). It is continuous but not smooth.
'''
print(sess.run(tf.nn.relu([-3,3,10])))
'''
[0.,3.,10]
'''
print(sess.run(tf.nn.relu6([-3,3,10])))
'''
define relu6() min(max(0,x),6)
[0.,3.,6]
'''
print(sess.run(tf.nn.sigmoid([-1,0.,1.])))
'''
[0.26894143   0.5   0.7310586]
'''
print(sess.run(tf.nn.tanh([-1,0.,1.])))
'''
[-0.76159418   0.   0.76159418]
'''
print(sess.run(tf.nn.softsign([-1,0.,1.])))
'''
x/(abs(x)+1)
[-0.5   0.   0.5]
'''
print(sess.run(tf.nn.softplus([-1,0.,1.])))
'''
log(exp(x)+1)
[0.31326166   0.69314718   1.31326163]
'''
print(sess.run(tf.nn.elu([-1,0.,1.])))
'''
exponential Linear Unit
[-0.63212055   0.   1.]
'''



