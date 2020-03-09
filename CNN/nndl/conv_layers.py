import numpy as np
from nndl.layers import *
import pdb



def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  pad = conv_param['pad']
  stride = conv_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the forward pass of a convolutional neural network.
  #   Store the output as 'out'.
  #   Hint: to pad the array, you can use the function np.pad.
  # ================================================================ #
  
 
  N,C,H,W = x.shape
  F,C,HH,WW= w.shape
  H_conv = int((H - HH+2*pad)/stride + 1)
  W_conv = int((W - HH+2*pad)/stride + 1)
  
  x_pad = np.pad(x,((0, 0), (0, 0), (pad,pad), (pad, pad)),mode = 'constant')
  
  out = np.zeros((N,F,H_conv,W_conv))
   
  for i in np.arange(N):
        for j in np.arange(F):
            for k in np.arange(H_conv):
                for m in np.arange(W_conv):
                    x_patch = x_pad[i,:,(k*stride):(k*stride+HH),(m*stride):(m*stride+HH)] 
                    
                    out[i,j,k,m] =np.sum(np.multiply(x_patch,w[j,:,:,:]))+b[j]
 
    
  #pdb.set_trace()
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #
    
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None

  N, F, out_height, out_width = dout.shape
  x, w, b, conv_param = cache
  
  stride, pad = [conv_param['stride'], conv_param['pad']]
  xpad = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), mode='constant')
  num_filts, _, f_height, f_width = w.shape

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the backward pass of a convolutional neural network.
  #   Calculate the gradients: dx, dw, and db.
  # ================================================================ #
  dw = np.zeros_like(w)
  db = np.zeros_like(b) 
  dx = np.zeros_like(xpad)
    
  for i in np.arange(N):
       for j in np.arange(F):
           for k in np.arange(out_height):
               for m in np.arange(out_width):
                   x_patch = xpad[i,:,(k*stride):(k*stride+f_height),(m*stride):(m*stride+f_height)] 
                   dw[j,:,:,:] += x_patch*dout[i,j,k,m]
                   db[j] +=dout[i,j,k,m]
                   dx[i,:,(k*stride):(k*stride+f_height),(m*stride):(m*stride+f_height)] += w[j,:,:,:] *dout[i,j,k,m]
   
  dx = dx[:,:,pad:-pad,pad:-pad] 
    
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ #

  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  
  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling forward pass.
  # ================================================================ #
  N,C,H,W = x.shape
  pool_height,pool_width,stride = [pool_param['pool_height'],pool_param['pool_width'],pool_param['stride']]
  H_out = int((H - pool_height)/stride +1)
  W_out = int((W- pool_width)/stride +1)
  out = np.zeros((N,C,H_out,W_out))
  for i in np.arange(N):
      for j in np.arange(C):
          for k in np.arange(H_out):
              for m in np.arange(W_out):
                  out[i,j,k,m] = np.max(x[i,j,(k*stride):(k*stride+pool_height),(m*pool_width):(m*stride+pool_width)])
  
  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  cache = (x, pool_param)
  return out, cache

def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the max pooling backward pass.
  # ================================================================ #
  N,C,H,W = x.shape
  _,_,H_out,W_out = dout.shape
  dx = np.zeros_like(x)
  for i in np.arange(N):
    for j in np.arange(C):
        for k in np.arange(H_out):
            for m in np.arange(W_out):
                x_patch = x[i,j,(k*stride):(k*stride+pool_height),(m*stride):(m*stride+pool_width)]
                max_Id = np.unravel_index(np.argmax(x_patch),(pool_height,pool_width))
                #pdb.set_trace()
                k_max = max_Id[0]+k*stride
                m_max = max_Id[1]+m*stride
                dx[i,j,k_max,m_max] += dout[i,j,k,m]
                
                
                
    

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 
  return dx

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.
  
  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features
    
  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm forward pass.x_spatial = x.transpose().reshape(C)
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N,C,H,W = np.shape(x)
  #pdb.set_trace()
  x_spatial = x.transpose(0,2,3,1).reshape(-1,C)
  
    
  out_spatial, cache = batchnorm_forward(x_spatial, gamma, beta, bn_param)
  out = out_spatial.reshape(N,H,W,C).transpose(0,3,1,2)

  

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return out, cache


def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.
  
  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass
  
  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  # ================================================================ #
  # YOUR CODE HERE:
  #   Implement the spatial batchnorm backward pass.
  #
  #   You may find it useful to use the batchnorm forward pass you 
  #   implemented in HW #4.
  # ================================================================ #
  N,C,H,W = np.shape(dout)
  dout_spatial =dout.transpose(0,2,3,1).reshape(-1,C)
  dx_spatial,dgamma,dbeta =batchnorm_backward(dout_spatial, cache)
  dx = dx_spatial.reshape(N,H,W,C).transpose(0,3,1,2)

  # ================================================================ #
  # END YOUR CODE HERE
  # ================================================================ # 

  return dx, dgamma, dbeta