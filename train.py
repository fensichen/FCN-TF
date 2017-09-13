import os
import scipy as scp
import numpy as np
import tensorflow as tf
import fcn8
import data_handler
import matplotlib.pyplot as plt

BATCH_SIZE          = 1
IMAGE_H             = 375
IMAGE_W             = 1242
IMAGE_C             = 3
num_classes         = 2


def visualize( data ):
    # Define 21 colors in RGB notation
  colors = []
  
  dim    = int( np.ceil( np.power( num_classes, 1./3 ) ) )
  for i in range( dim ):
    for j in range( dim ):
        for k in range( dim ):
            colors.append( [i*int(255/dim), j*int(255/dim), k*int(255/dim)] )
    
  colors = np.array( colors )
  vis    = np.zeros( [data.shape[1], data.shape[2], 3] )
  
  for i in range( data.shape[1] ):
    for j in range( data.shape[2] ):
        print "i,j", i, j
        print "data[0,i,j]", data[0,i,j]
        vis[i,j,:] = colors[ data[0,i,j] ]
  
  plt.imshow( vis )
  plt.show()


fcn_net             = fcn8.FCN( num_classes = 2, batch_size = BATCH_SIZE )

# construct model
images              = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_H, IMAGE_W, IMAGE_C] )
labels              = tf.placeholder(tf.int64  , [BATCH_SIZE, IMAGE_H, IMAGE_W, num_classes] )

fcn_net.build_seg_net(images)    
# set gpu configuration
conf                = tf.ConfigProto(gpu_options = tf.GPUOptions( allow_growth = True ) )

# load data 
hnd_data_train      = data_handler.DataHandler()
hnd_data_train.get_file_list('/home/fensi/nas/KITTI_ROAD/train.txt', '/home/fensi/nas/KITTI_ROAD/label.txt')
hnd_data_test       = data_handler.DataHandler()
hnd_data_test.get_file_list('/home/fensi/nas/KITTI_ROAD/test.txt')

# loss_op need to be called before train_op, please see the definition of train_op in fcn8.py 
fcn_loss            = fcn_net.loss_op( logits = fcn_net.result, labels = labels )
fcn_op              = fcn_net.train_op()

num_steps           = 200
with tf.Session( config = conf) as sess:
    init     = tf.global_variables_initializer()
    sess.run(init)

    # training
    for step in range( 1,  num_steps + 1 ):
        num                        = hnd_data_train.num()
        avg_avg_acc                = np.zeros( num )
        for start in range( 0, num-(num%BATCH_SIZE), BATCH_SIZE) :
          end                      = start + BATCH_SIZE
          image_batch, label_batch = hnd_data_train.load_data( start, end )
          feed                     = { images : image_batch, labels : label_batch }
          _, loss,res              = sess.run( [fcn_op, fcn_loss, fcn_net.result], feed_dict=feed)

          # Assuming res is shape B x H x W x C and label_batch is shape B x H x W        
          diff                     = np.argmax( res, axis = 3 ) - np.argmax( label_batch, axis = 3 )
          accuracy                 = (diff == np.zeros_like( diff ))
          accuracy                 = np.reshape( accuracy, [accuracy.shape[0], -1] )
          avg_acc                  = np.average( accuracy, axis=1 )
          avg_avg_acc[start : end] = avg_acc
        
        print "average accuray: ", np.average( avg_avg_acc )
      
        # testing
        num = hnd_data_test.num()
        for start in range( 1, num - (num % BATCH_SIZE), BATCH_SIZE ):
            end                    = start + BATCH_SIZE
            image_batch            = hnd_data_test.load_data( start, end, is_training = False)
            feed                   = { images : image_batch } 
            res                    = sess.run( fcn_net.result, feed_dict=feed )
            print res
            res                    = res + [ 123.68, 116.779, 103.939]
            visualize( res )