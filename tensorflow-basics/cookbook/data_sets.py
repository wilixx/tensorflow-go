'''
Created on Mar 4, 2018

@author: Dr.Guo
'''
import tensorflow as tf
import numpy as np
from PIL.ImageChops import constant
from jinja2.runtime import identity
from sklearn import datasets
from tensorflow.contrib.factorization.examples.mnist import MnistTest
'''
Iris data
'''
iris=datasets.load_iris()
print(len(iris.data))
print(len(iris.target))
print(iris.target[0])
print(set(iris.target))

'''
Birth data
'''
import requests
birthdata_url=""
birth_file=requests.get(birthdata_url)
birth_data=birth_file.text.split('\'r\n')[5:]
birth_header=[x for x in birth_data[0].split('') if len(x(0)>=1)]
birth_data=''
print(len(birth_data))
print(len(birth_data[0]))

'''
Boston Housing data
'''
import requests

'''
MNIST handwritting data(Mixed National Insititute of Standard and Tech)
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("Mnist_data/", one_hot=True)
print(len(mnist.train.images)
print(len(mnist.test.images)
print(len(mnist.validation.images)
print(len(mnist.train.labels[1:])

'''
Spam-ham text data
'''
import requests

pass 
pass


'''
movie review data
'''
import requests
import io
import tarflie

movie_data_url="http://www.cs.cornell.edu/people/pabo/movie-review-data/"

'''
CIFAR-10 image data
'''
import requests
import io
import tarflie


'''
The works of Shakespeare text data
'''
import requests
import io
import tarflie

'''
English-German sentence translation data
'''
import requests
import io
import tarflie














