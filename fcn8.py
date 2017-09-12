import tensorflow as tf
import numpy as np
import os
import sys

vgg_16_npy_path = 'vgg16.npy' #VGG16 net: ['conv5_1', 'fc6', 'conv5_3', 'fc7', 'fc8', 'conv5_2', 'conv4_1', 'conv4_2', 'conv4_3', 'conv3_3', 'conv3_2', 'conv3_1', 'conv1_1', 'conv1_2', 'conv2_2', 'conv2_1']
kittiSeg_pretrained_path = '/home/fensi/nas/KittiSeg_pretrained/model.ckpt-15999.npy'


class FCN(object):
    # load model
    def __init__(self, model_path = kittiSeg_pretrained_path, num_classes = 2, batch_size = 16 ):
        
        self.model        = np.load( model_path, encoding='latin1').item() 
        self.num_classes  = num_classes
        self.batch_size   = batch_size
        

    def get_weight(self, name):
        """
        load weight from pretrained model (npy) 
        """
        with tf.variable_scope( name ): #Q: Do we need this line ? 
            init_   = tf.constant_initializer( value = self.model[name], dtype = tf.float32)
            shape_  = self.model[name].shape
            weight_ = tf.get_variable(name, initializer = init_, shape = shape_ )
            return weight_

    def get_bias(self, name):
        """
        load bias from pretrained model
        """
        name    = name +"Bias"
        init_   = tf.constant_initializer( value = self.model[name ], dtype = tf.float32)
        shape_  = self.model[name ].shape
        bias_   = tf.get_variable(name , initializer = init_, shape = shape_ )
        return bias_

    def conv2d(self, x, name):
        """
        Perform convolution using pretrained weight/bias  
        """
        with tf.variable_scope( name ): #Q:Do we need to add this line?
            W       = self.get_weight(name)
            b       = self.get_bias(name)
            x       = tf.nn.conv2d( x, W, [1, 1, 1, 1], padding = 'SAME')
            x       = tf.nn.bias_add(x, b)
            return tf.nn.relu(x)

    def max_pool(self, x, name):
        """
        Maximun pooling with kernel size = 2, stride = 2 
        """
        return tf.nn.max_pool( x, ksize= [1, 2, 2, 1],  strides = [1, 2, 2, 1], padding = 'SAME')

    def get_weight_fc_reshape(self, name, shape_):
        """
        reshape the pretrained fc weight into new shape (shape_)
        """
        W       = self.model[name]
        W       = W.reshape(shape_)
        init_   = tf.constant_initializer( value = W, dtype = tf.float32 )
        weight_ = tf.get_variable( name, initializer = init_, shape = shape_)
        return weight_

   
    def fc(self, x, name ):
        """
        Transform fully connected layer to convolution layer by first reshaping the (pretrained) weight kernel, 
        then convoluting the input tensor using the reshaped kernel 
        """
        with tf.variable_scope( name ) as scope:
            if name == 'fc6':
                W = self.get_weight_fc_reshape( name, [7, 7,  512, 4096])
            elif name == 'score_fr':
                W = self.get_weight_fc_reshape( name, [1, 1, 4096, self.num_classes])
            else:
                W = self.get_weight_fc_reshape( name, [1, 1, 4096, 4096])

            x     = tf.nn.conv2d( x, W, [1, 1, 1, 1], padding = 'SAME' )
            b     = self.get_bias( name  )
            x     = tf.nn.bias_add( x, b )
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

    def upsample_layer(self, bottom, name, upscale_factor, shape):
        """
        The spatial extent of the output map can be optained from the fact that (upscale_factor -1) pixles are inserted between two successive pixels
        """

        kernel_size  = self.get_kernel_size( upscale_factor )
        stride       = upscale_factor
        strides      = [1, stride, stride, 1]
        # data tensor: 4D tensors are usually: [BATCH, Height, Width, Channel]
        n_channels   = bottom.get_shape()[-1].value

        with tf.variable_scope(name) as scope:
            # shape of the bottom tensor
            if shape is None:
                in_shape     = tf.shape(bottom) 
                print "in_shape", in_shape.get_shape().as_list           
                h            = ( ( in_shape[1] - 1 ) * stride ) + 1
                w            = ( ( in_shape[2] - 1 ) * stride ) + 1
                new_shape    = [ in_shape[0], h, w, n_channels]

            else:
                new_shape    = [shape[0], shape[1], shape[2], shape[3]]

            output_shape = tf.stack( new_shape )
            filter_shape = [kernel_size, kernel_size, n_channels, n_channels ] # Q: why "n_channels" filter? 
            weights_     = self.get_upsample_filter(filter_shape, upscale_factor )
            deconv       = tf.nn.conv2d_transpose(bottom, weights_, output_shape, strides = strides, padding= 'SAME')
            return deconv

    def score_layer ( self, bottom, name):
        """
        append 1x1 convolution with channel dimension for each class at each of the coarse output location
        """
        with tf.variable_scope( name ) as scope:
            in_channels  = bottom.get_shape()[3].value # the channel of input tensor
            shape        = [ 1, 1, in_channels, self.num_classes ] # define the shape of convolution kernel
            W            = self.get_weight(name)
            b            = self.get_bias(name)
            conv         = tf.nn.conv2d( bottom, W, [ 1, 1, 1, 1], padding = 'SAME')
            x            = tf.nn.bias_add( conv, b ) 
            return x 

    def build_seg_net(self, img ):
        """
        Build Seg-Net using pre-trained weight parameters, followed by a deconvolution layer to bilinearly upsample the coarse outputs to pixel-dense outputs. 
        """

        # Q: convert rgb to bgr (?) 
        self.conv1_1     = self.conv2d(img,            "conv1_1")
        self.conv1_2     = self.conv2d(self.conv1_1,   "conv1_2")
        self.pool1       = self.max_pool(self.conv1_2, "pool1"  )   
       
        self.conv2_1     = self.conv2d(self.pool1,     "conv2_1")
        self.conv2_2     = self.conv2d(self.conv2_1,   "conv2_2")
        self.pool2       = self.max_pool(self.conv2_2, "pool2"  )
       
        self.conv3_1     = self.conv2d(self.pool2,     "conv3_1")
        self.conv3_2     = self.conv2d(self.conv3_1,   "conv3_2")
        self.conv3_3     = self.conv2d(self.conv3_2,   "conv3_3")
        self.pool3       = self.max_pool(self.conv3_3, "pool3"  )

        self.conv4_1     = self.conv2d(self.pool3,     "conv4_1")
        self.conv4_2     = self.conv2d(self.conv4_1,   "conv4_2")    
        self.conv4_3     = self.conv2d(self.conv4_2,   "conv4_3")
        self.pool4       = self.max_pool(self.conv4_3, "pool4"  )

        self.conv5_1     = self.conv2d(self.pool4,     "conv5_1")
        self.conv5_2     = self.conv2d(self.conv5_1,   "conv5_2")
        self.conv5_3     = self.conv2d(self.conv5_2,   "conv5_3")
        self.pool5       = self.max_pool(self.conv5_3, "pool5"  )

        # fully conv 
        self.fc6         = self.fc(self.pool5,         "fc6"    )
        self.fc7         = self.fc(self.fc6,           "fc7"    )
        self.score_fr    = self.fc(self.fc7,           "score_fr" )
        # upsampling : strided convolution
        self.score_pool4 = self.score_layer( self.pool4, "score_pool4")
        self.upscore2    = self.upsample_layer(self.score_fr, "upscore2", 2, self.score_pool4.get_shape().as_list() ) # Q: why not conv7 as in the paper?  
        self.fuse_pool4  = tf.add( self.upscore2, self.score_pool4)
       
        self.score_pool3 = self.score_layer( self.pool3, "score_pool3")
        self.upscore4    = self.upsample_layer( self.fuse_pool4, "upscore4", 2, self.score_pool3.get_shape().as_list() )
        self.fuse_pool3  = tf.add( self.upscore4, self.score_pool3)
       
        imgshape         = img.get_shape().as_list()
        target_shape     = [ self.batch_size, imgshape[1], imgshape[2], self.num_classes ]
        self.upscore32   = self.upsample_layer( self.fuse_pool3, "upscore32", 8, target_shape ) # 8x upsampled prediction
       
        self.result      = self.upscore32 
        

    def train_op(self):
        optimizer        = tf.train.AdamOptimizer( 1e-3 )
        train_op         = optimizer.minimize( self.loss_op)
        return train_op
        

    def loss_op(self, logits, labels):
        """     
        Args: 
            logits: tensor, float        - [ batch_size, width, height, num_classes ]
            labels: labels tensor, int32 - [ batch_size, width, height, num_classes ]
        Returns
            loss: 
        """
        # If logits is B x D x C, then sparse labels: B x D, normal labels: B x D x C
        logits = tf.reshape( logits, [self.batch_size, -1, self.num_classes] ) 
        # --> to make it as BxDxC (?), where D = width * height * channels  
        
        # Make sure logits has dimension D+1 (for labels dimension D)
        print "logits.shape", logits.get_shape().as_list()
        print "labels.shape", labels.shape
        #loss_op         = tf.nn.sparse_softmax_cross_entropy_with_logits( logits = logits, labels = labels )
        loss_op         = tf.nn.softmax_cross_entropy_with_logits( logits = logits, labels = labels )
        
        # If we have B x D x 1
        loss_op         = tf.reduce_sum( loss_op, axis = 1 )
        # --> B x 1
        loss_op         = tf.reduce_mean( loss_op )
        # --> 1
        self.loss_op    = loss_op
        return loss_op



