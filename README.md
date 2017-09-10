# FCN-TF

This is a tensorflow implementation of the paper [Fully Convolutional Networks for Image Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)



## Fractional strided Convolution ( transposed convolution )

Normal convolution convolves the image depending on the parameters ( stride, kernel size, padding) to reduce the input image. 
Transposed convolution upsample the input image. 

If we define a bilinear upsampling kernel and perform fractionally strided covolution on the image, we will get an upsampled output. We will use bilinear up-samplling kernel as initialization, then the network can learn a more suitable kernel during backpropagation.

The factor of up-sampling is equal to the stride of transposed convolution. The kernel of the upsampling operation is determined as  2 * factor - factor % 2 

To get more information about transposed convolution, we refer to these guides
[1](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)
[2](https://github.com/vdumoulin/conv_arithmetic)
[3](http://cv-tricks.com/image-segmentation/transpose-convolution-in-tensorflow/)

### USAGE 
python main.py to test the implementation