import tensorflow as tf
import numpy as np
import os
import sys

vgg_16_npy_path = 'vgg16.npy'

class FCN(object):
    # load model
    def __init__(self, **kwargs):
        #['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2', 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
        self.model = np.load( vgg_16_npy_path, encoding='latin1').item() 
        print "load vgg_16.npy" 

        self.num_classes  = 1000 

    def get_weight(self, name):
        
        init_   = tf.constant_initializer( value = self.model[name][0], dtype = tf.float32)
        shape_  = self.model[name][0].shape
        weight_ = tf.get_variable(name, initializer = init_, shape = shape_ )
        return weight_

    def get_weight_fc_reshape(self, name, shape_, num_classes = None ):
        """
        """
        W       = self.model[name][0]
        W       = W.reshape(shape_)

        init_   = tf.constant_initializer( value = W, dtype = tf.float32 )
        weight_ = tf.get_variable( name, initializer = init_, shape = shape_)
        return weight_

    def get_bias(self, name):

        init_   = tf.constant_initializer( value = self.model[name][1], dtype = tf.float32)
        shape_  = self.model[name][1].shape
        bias_   = tf.get_variable(name + "Bias", initializer = init_, shape = shape_ )
        return bias_

    def conv2d(self, x, name):
        with tf.variable_scope( name ) as scope: # (Q:) why need to add this line?
            W       = self.get_weight(name)
            b       = self.get_bias(name)
            x       = tf.nn.conv2d( x, W, [1, 1, 1, 1], padding = 'SAME')
            x       = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

    def max_pool(self, x, name):
        return tf.nn.max_pool( x, ksize= [1, 2, 2, 1],  strides = [1, 2, 2, 1], padding = 'SAME')

    # transform FC layer to convolution 
    def fc(self, x, name, num_classes = None ):
        """
        """

        if name == 'fc6':
            W = self.get_weight_fc_reshape( name, [7, 7,  512, 4096])
        elif name == 'fc8':
            W = self.get_weight_fc_reshape( name, [1, 1, 4096, num_classes])
            #W = tf.Variable( tf.random_normal( [1, 1, 4096, num_classes], stddev = 0.01 ) )
        else:
            W = self.get_weight_fc_reshape( name, [1, 1, 4096, 4096])

        x     = tf.nn.conv2d( x, W, [1, 1, 1, 1], padding = 'SAME' )

        if name == 'fc8':
            #b     = tf.Variable( tf.random_normal( [ num_classes], stddev = 0) )
            b     = self.get_bias( name )
        else:
            b     = self.get_bias( name )
    
        x     = tf.nn.bias_add( x, b)
        
        return tf.nn.relu( x )

    def get_kernel_size(self, factor):
        """
        Find the kernel size given the desired factor of upsampling
        """
        return 2 * factor - factor % 2

    def get_upsample_filter(self, filter_shape, upscale_factor):
        """
        Make a 2D bilinear kernel 
        """
        ### filter_shape is [ width, height, num_in_channel, num_out_channel ]
        kernel_size = filter_shape[1]
        ### center location of the filter for which value is calculated
        if kernel_size % 2 == 1:
            center_location = upscale_factor -1
        else:
            center_location = upscale_factor - 0.5 # e.g.. 

        bilinear_grid = np.zeros([ filter_shape[0], filter_shape[1] ] )
        for x in range( filter_shape[0] ):
            for y in range ( filter_shape[1] ):
                ### Interpolation Calculation
                value         = ( 1 - abs( (x - center_location)/upscale_factor ) ) * ( 1 - abs( (y - center_location)/upscale_factor ) )
                bilinear_grid[x,y] = value

        weights = np.zeros( filter_shape )

        for i in range( filter_shape[2]):
            weights[:,:,i,i] = bilinear_grid

        init             = tf.constant_initializer( value = weights, dtype = tf.float32 )
        bilinear_weights = tf.get_variable( name = "deconv_bilinear_filter", initializer = init, shape = weights.shape ) 

        return bilinear_weights

    def upsample_layer(self, bottom, name,  upscale_factor = 32):
        """
        * The spatial extent of the output map can be optained from the fact that (upscale_factor -1) pixles are inserted between two successive pixels
        * 
        """

        kernel_size  = self.get_kernel_size( upscale_factor )
        stride       = upscale_factor
        strides      = [1, stride, stride, 1]
    # data tensor: 4D tensors are usually: [BATCH, Height, Width, Channel]
        n_channels   = bottom.get_shape()[-1].value

        with tf.variable_scope(name):
            in_features  = bottom.get_shape()[3].value
            # shape of the bottom tensor
            in_shape     = tf.shape(bottom)
            print "in_shape", bottom.get_shape().as_list()
            h            = ( ( in_shape[1] - 1 ) * stride ) + 1
            w            = ( ( in_shape[2] - 1 ) * stride ) + 1
            new_shape    = [ in_shape[0], h, w, n_channels]

            output_shape = tf.stack( new_shape )

            filter_shape = [kernel_size, kernel_size, n_channels, n_channels ] # number of input channel, [3] the number of output channel
            print filter_shape
            weights_     = self.get_upsample_filter(filter_shape, upscale_factor )
            print weights_.get_shape().as_list()
            deconv       = tf.nn.conv2d_transpose(bottom, weights_, output_shape, strides = strides, padding= 'SAME')
        return deconv

    # build the VGG model using
    def build_vgg_net(self, img ):
        """
        Build VGG using pre-trained weight parameters
        """

        # Q: convert rgb to bgr (?) 


        self.conv1_1 = self.conv2d(img,            "conv1_1")
        self.conv1_2 = self.conv2d(self.conv1_1,   "conv1_2")
        self.pool1   = self.max_pool(self.conv1_2, "pool1"  )

        self.conv2_1 = self.conv2d(self.pool1,     "conv2_1")
        self.conv2_2 = self.conv2d(self.conv2_1,   "conv2_2")
        self.pool2   = self.max_pool(self.conv2_2, "pool2"  )

        self.conv3_1 = self.conv2d(self.pool2,     "conv3_1")
        self.conv3_2 = self.conv2d(self.conv3_1,   "conv3_2")
        self.conv3_3 = self.conv2d(self.conv3_2,   "conv3_3")
        self.pool3   = self.max_pool(self.conv3_3, "pool3"  )

        self.conv4_1 = self.conv2d(self.pool3,     "conv4_1")
        self.conv4_2 = self.conv2d(self.conv4_1,   "conv4_2")
    
        self.conv4_3 = self.conv2d(self.conv4_2,   "conv4_3")
        self.pool4   = self.max_pool(self.conv4_3, "pool4"  )

        self.conv5_1 = self.conv2d(self.pool4,     "conv5_1")
        self.conv5_2 = self.conv2d(self.conv5_1,   "conv5_2")
        self.conv5_3 = self.conv2d(self.conv5_2,   "conv5_3")
        self.pool5   = self.max_pool(self.conv5_3, "pool5"  )

        self.fc6     = self.fc(self.pool5,         "fc6"    )
        self.fc7     = self.fc(self.fc6,           "fc7"    )

        self.score_fr= self.fc(self.fc7,           "fc8", self.num_classes )

        # upsampling : strided convolution
        self.result  = self.upsample_layer(self.score_fr, "up_sample", upscale_factor = 32)






