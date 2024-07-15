import numpy as np
from CNN.utils import *

#####################################################
############### Backward Operations #################
#####################################################
        
def convolutionBackward(dconv_prev, conv_in, filt, s):
    '''
    Backpropagation through a convolutional layer. 
    '''
    (n_f, n_c, f, _) = filt.shape
    (_, orig_dim, _) = conv_in.shape
    ## initialize derivatives
    dout = np.zeros(conv_in.shape) 
    dfilt = np.zeros(filt.shape)
    dbias = np.zeros((n_f,1))
    for curr_f in range(n_f):
        # loop through all filters
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                # loss gradient of filter (used to update the filter)
                dfilt[curr_f] += dconv_prev[curr_f, out_y, out_x] * conv_in[:, curr_y:curr_y+f, curr_x:curr_x+f]
                # loss gradient of the input to the convolution operation (conv1 in the case of this network)
                dout[:, curr_y:curr_y+f, curr_x:curr_x+f] += dconv_prev[curr_f, out_y, out_x] * filt[curr_f] 
                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1
        # loss gradient of the bias
        dbias[curr_f] = np.sum(dconv_prev[curr_f])
    
    return dout, dfilt, dbias


def meanpoolBackward(dpool, orig, f, s):
    '''
    Backpropagation through an average pooling layer. The gradients are evenly distributed back to the original pooling region.
    '''
    (n_c, orig_dim, _) = orig.shape
    (dpool_dim, _, _) = dpool.shape  # Get the dimensions of dpool

    dout = np.zeros(orig.shape)

    for curr_c in range(n_c):
        curr_y = out_y = 0
        while curr_y + f <= orig_dim:
            curr_x = out_x = 0
            while curr_x + f <= orig_dim:
                if out_y < dpool_dim and out_x < dpool_dim:  # Ensure we do not go out of bounds
                    # Distribute the gradient to all the elements in the pooling region
                    gradient = dpool[curr_c, out_y, out_x] / (f * f)
                    dout[curr_c, curr_y:curr_y + f, curr_x:curr_x + f] += gradient

                curr_x += s
                out_x += 1
            curr_y += s
            out_y += 1

    return dout


