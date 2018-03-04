'''
Created on Mar 4, 2018

@author: Dr.Guo
'''
import tensorflow as tf

x = tf.constant(35, name='x')
y = tf.Variable(x + 5, name='y')
# x = tf.constant([35, 40, 45], name='x')
# y = tf.Variable(x + 5, name='y')

print(y)
print(x)
model = tf.global_variables_initializer()

with tf.Session() as session:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("basic", session.graph)
    model = tf.global_variables_initializer()
    session.run(model)
    print(session.run(y))
    # session.run(model)
    # print("x=:",session.run(x))
    # print("y=:",session.run(y))