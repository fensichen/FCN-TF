import os
import scipy as scp
import scipy.misc
import tensorflow as tf
import numpy as np
#import fcn
import fcn8
import matplotlib.pyplot as plt

num_class = 2

def visualize( data ):
    # Define 21 colors in RGB notation
  colors = []
  
  dim    = int( np.ceil( np.power( num_class, 1./3 ) ) )
  for i in range( dim ):
    for j in range( dim ):
        for k in range( dim ):
            colors.append( [i*int(255/dim), j*int(255/dim), k*int(255/dim)] )
    
  colors = np.array( colors )
  vis    = np.zeros( [data.shape[1], data.shape[2], 3] )
  
  for i in range( data.shape[1] ):
    for j in range( data.shape[2] ):
        vis[i,j,:] = colors[ data[0,i,j] ]
  
  plt.imshow( vis )
  plt.show()
  

BATCH=1

img_in  = scp.misc.imread('/home/fensi/nas/data_road/training/image_2/um_000000.png')

fcn_net = fcn8.FCN()
fcn_net.num_classes = num_class
image   = tf.placeholder(tf.float32,  [BATCH, img_in.shape[0], img_in.shape[1], img_in.shape[2]] )
fcn_net.build_vgg_net(image)    

saver = tf.train.Saver()

vars = tf.trainable_variables()
varnames = [v.name for v in vars]
print varnames
#exit(0)

with tf.Session() as sess:
    #saver.restore(sess,"/home/fensi/nas/KittiSeg_pretrained/model.ckpt-15999")
    init     = tf.global_variables_initializer()
    sess.run(init)
    img_in   = np.expand_dims( img_in, axis = 0)
    res = sess.run( fcn_net.result, feed_dict={image : img_in } )
    print res.shape
    out = np.argmax(res, axis = 3) 
    print out
    visualize( out )
