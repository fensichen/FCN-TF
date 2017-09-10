import numpy
from tensorflow.python import pywrap_tensorflow

def conversion():
    translate = {}
    translate['conv1_1/filter']      = 'conv1_1'
    translate['conv1_1/biases']      = 'conv1_1Bias'
    translate['conv1_2/filter']      = 'conv1_2'
    translate['conv1_2/biases']      = 'conv1_2Bias'
    translate['conv2_1/filter']      = 'conv2_1'
    translate['conv2_1/biases']      = 'conv2_1Bias'
    translate['conv2_2/filter']      = 'conv2_2'
    translate['conv2_2/biases']      = 'conv2_2Bias'
    translate['conv3_1/filter']      = 'conv3_1'
    translate['conv3_1/biases']      = 'conv3_1Bias'
    translate['conv3_2/filter']      = 'conv3_2'
    translate['conv3_2/biases']      = 'conv3_2Bias'
    translate['conv3_3/filter']      = 'conv3_3'
    translate['conv3_3/biases']      = 'conv3_3Bias'
    translate['conv4_1/filter']      = 'conv4_1'
    translate['conv4_1/biases']      = 'conv4_1Bias'
    translate['conv4_2/filter']      = 'conv4_2'
    translate['conv4_2/biases']      = 'conv4_2Bias'
    translate['conv4_3/filter']      = 'conv4_3'
    translate['conv4_3/biases']      = 'conv4_3Bias'
    translate['conv5_1/filter']      = 'conv5_1'
    translate['conv5_1/biases']      = 'conv5_1Bias'
    translate['conv5_2/filter']      = 'conv5_2'
    translate['conv5_2/biases']      = 'conv5_2Bias'
    translate['conv5_3/filter']      = 'conv5_3'
    translate['conv5_3/biases']      = 'conv5_3Bias'
    translate['fc6/weights']         = 'fc6'
    translate['fc6/biases']          = 'fc6Bias'
    translate['fc7/weights']         = 'fc7'
    translate['fc7/biases']          = 'fc7Bias'
    translate['score_fr/weights']    = 'score_fr'
    translate['score_fr/biases']     = 'score_frBias'
    translate['upscore2/up_filter']  = 'upscore2'     
    translate['upscore4/up_filter']  = 'upscore4'
    translate['upscore32/up_filter'] = 'upscore32'
    translate['score_pool3/weights'] = 'score_pool3' 
    translate['score_pool3/biases']  = 'score_pool3Bias'
    translate['score_pool4/weights'] = 'score_pool4'
    translate['score_pool4/biases']  = 'score_pool4Bias'
    return translate



filename = '/home/fensi/nas/KittiSeg_pretrained/model.ckpt-15999'
newfilename = '/home/fensi/nas/KittiSeg_pretrained/model.ckpt-15999.npy'

reader = pywrap_tensorflow.NewCheckpointReader( filename )
var_map = reader.get_variable_to_shape_map()

dict = {}
translate = conversion()

for key in var_map:
    if "Optimizer" in key:
        continue
    if "upscore" in key:
        continue

    name = translate[key]
    value = reader.get_tensor( key )
    dict[name] = value

np_dict = numpy.array( dict )
numpy.save( newfilename, np_dict )

