#!/usr/bin/env python

from time import time
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops

from ipdb import set_trace as st

from scipy.misc import face
import numpy as np

from tftransform import *
import random

mode = 'perf'

assert mode in ['perf', 'test']


if mode == 'perf':

    #GD
    #tfimpl: max bs = 4, 39ms
    #native: max bs = 32, 30ms
    
    #ADAM
    #tfimpl: max bs 4, 43.5 ms
    #native: max bs 14, 30 ms
    B = 32
    S = 64
    C = 64
else:
    B = 8
    S = 8
    C = 8

if mode == 'test':
    x = np.random.randn(B, S, S, S, C).astype(np.float32)
    #print('fixed x')
    #x = np.array([[[1,2],[3,4]],[[5,6],[7,8]]]).astype(np.float32)[np.newaxis,:,:,:,np.newaxis]
    #x = np.concatenate([x,x], axis = 0)
    #wtf are we getting as the outputs?

    x_0 = tf.Variable(x, trainable = True)
    x_1 = tf.Variable(x, trainable = True)
    
else:
    x_0 = tf.get_variable("x_0", [B, S, S, S, C], dtype=tf.float32, initializer=tf.random_normal_initializer)
    x_1 = tf.get_variable("x_1", [B, S, S, S, C], dtype=tf.float32, initializer=tf.random_normal_initializer)

tf_grid = tf.tile(tf.expand_dims(tf.constant(numpy_grid(S)), axis = 0), [B, 1, 1, 1, 1])

R = tf.Variable(rand_trans(B), trainable = True)

#rotate the grid and compute the loss
tf_grid_flat = tf.reshape(tf_grid, (B, -1, 3))
tf_grid_flat_transformed = tf.matmul(tf_grid_flat, R)
tf_grid_transformed = tf.reshape(tf_grid_flat_transformed, tf_grid.shape)

#import sys
#assert len(sys.argv) > 1

rot_x_0 = tf_grid_sample(x_0, tf_grid_transformed, S, False)
rot_x_1 = tf_grid_sample(x_1, tf_grid_transformed, S, True)

loss_0 = tf.reduce_mean(tf.abs(rot_x_0))
loss_1 = tf.reduce_mean(tf.abs(rot_x_1))
diff = tf.reduce_max(tf.abs(rot_x_0-rot_x_1))

optop_0 = tf.train.GradientDescentOptimizer(0.1).minimize(loss_0)
optop_1 = tf.train.GradientDescentOptimizer(0.1).minimize(loss_1)
# optop_0 = tf.train.AdamOptimizer(0.1).minimize(loss_0)
# optop_1 = tf.train.AdamOptimizer(0.1).minimize(loss_1)

g_0 = tf.gradients(loss_0, [x_0, tf_grid_transformed])
g_1 = tf.gradients(loss_1, [x_1, tf_grid_transformed])
gdiffs = tf.reduce_max(tf.abs(g_0[0]-g_1[0])) 
gdiffg = tf.reduce_max(tf.abs(g_0[1]-g_1[1]))

sess = tf.Session()

print('initializing session')
    
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

if mode == 'perf':
    #timing...
    def runop(x, N = 100):
        sess.run(x)
        t0 = time()    
        for i in range(N):
            sess.run(x)        
        return (time()-t0)/N

    print('time native is', runop(optop_1, 10))
    #print('time tfimpl is', runop(optop_0))

else:
    ro_0, ro_1, out_0, out_1, diff = sess.run([rot_x_0, rot_x_1, loss_0, loss_1, diff])
    og_0, og_1, gdiffs, gdiffg = sess.run([g_0, g_1, gdiffs, gdiffg])
    #out_0, out_1 = sess.run([rot_x_0, rot_x_1])

    #st()
    #loss_, _ = sess.run([loss, optop])
    
    #print('output 0 is', ro_0, out_0)
    #print('output 1 is', ro_1, out_1)
    print('diff is', diff)

    #print('og0 is', og_0)
    #print('og1 is', og_1)
    print('gdiff is', gdiffs, gdiffg)
