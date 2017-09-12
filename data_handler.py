import tensorflow as tf
import numpy as np
import os, sys
import random
import scipy as scp
import scipy.misc
import matplotlib.pyplot as plt

num_classes = 2

class DataHandler(object):
    def __init__(self):
        self.image_list     = []
        self.label_list     = []

    def num( self ):
        return len(self.image_list)

    def get_file_list(self, path_image, path_label = None):
        image_list          = [line.strip().split(' ')[0] for line in open( path_image )]
        self.image_list     = self.image_list + image_list 

        if path_label is not None:
            label_list          = [line.strip().split(' ')[0] for line in open( path_label )]
            self.label_list     = self.label_list + label_list

        
    def shuffle(self, filenames):
        self.perm           = range( len(filenames) )
        random.shuffle( self.perm ) # in-place shuffle 

    def load_data(self, start, end, is_training = True ):
        """
        Args: 
            start index
            end index
        Return:
            image array
            label array
        """
        
        batch_size          = end - start
        # Create data array
        probe_filename      = tf.read_file( self.image_list[0])
        probe               = scp.misc.imread( self.image_list[0] )
        HEIGHT              = probe.shape[0]
        WIDTH               = probe.shape[1]
        CHANNEL             = probe.shape[2]
        # shuffle       
        self.shuffle( self.image_list )
        data                = np.zeros( (batch_size, HEIGHT, WIDTH, CHANNEL) )
        labels              = np.zeros( (batch_size, HEIGHT, WIDTH, num_classes ) )

        VGG_MEAN            = [ 123.68, 116.779, 103.939] # RGB
        background          = [ 255,    0      ,   0    ]
        road                = [ 255,    0      , 255    ]

       
        #tmp = np.ones([ probe.shape[0], probe.shape[1], num_classes ])

        for index in range( batch_size ):
            
            img             = scp.misc.imread( self.image_list[ self.perm[ start + index ] ] )
            #print "loading ", self.image_list[ self.perm[ start + index ] ]
            if ( img.shape == probe.shape ) == False:
                img = np.resize( img, probe.shape )

            if is_training == True:
                label       = scp.misc.imread( self.label_list[ self.perm[ start + index ] ] )
                if ( label.shape == probe.shape ) == False:
                    label = np.resize( label, probe.shape )
               
                bg              = np.all( label == background, axis = 2)
                rd              = np.all( label == road, axis = 2)
                tmp             = np.ones( [ bg.shape[0], bg.shape[1], num_classes ])
                tmp[:,:,0]      = bg
                tmp[:,:,1]      = rd 
                labels[index]   = tmp
                
            #label           = np.expand_dims(label, axis=2)
            # Subtract average of each color channel to center the data around zero mean for each channel (R,G,B). 
            # This typically hels the network to learn faster since gradients act uniformly for each channel
            
            img[:,:,0]      = img[:,:,0] - VGG_MEAN[0]
            img[:,:,1]      = img[:,:,1] - VGG_MEAN[1]
            img[:,:,2]      = img[:,:,2] - VGG_MEAN[2]
            data[index]     = img
            
        if is_training == False:
            return data
        else:
            return data, labels

    #def generate_image_and_label_batch(self, image, label, batch_size):



if __name__ == '__main__':
    data_handler = DataHandler()
    data_handler.get_file_list( '/home/fensi/nas/KITTI_ROAD/train.txt', '/home/fensi/nas/KITTI_ROAD/label.txt' )
    data_handler.load_data(1,10)