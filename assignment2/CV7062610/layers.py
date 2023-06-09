from builtins import range
import numpy as np



def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    dim_size = x[0].shape

    # (N, D) x (D, M) + (1, M)
    out = x.reshape(x.shape[0], np.prod(dim_size)).dot(w)
    out += b.reshape(1, -1) 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    input_shape = x.shape
    # reshape the input from the cache
    x = x.reshape((input_shape[0], -1))
    
    # Out derivation w.r.t x
    dx = np.dot(dout, w.T).reshape(input_shape)
    # Out derivation w.r.t W
    dw = np.dot(x.T, dout)
    # Out derivation w.r.t b
    db = np.sum(dout, axis=0)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****u
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out = np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = np.copy(dout)
    dx[x<=0] = 0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input. 
        

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x_pad, w, b, conv_param)
    """
    out = None
    x_pad = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    stride = conv_param['stride']
    pad = conv_param['pad']

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    H_ = (H + 2 * pad - HH) // stride + 1 
    W_ = (W + 2 * pad - WW) // stride + 1


    # Initialize the output array
    out = np.zeros((N, F, H_, W_))

    # pad the input 
    x_pad = np.pad(x, ((0,0), (0,0),(pad, pad), (pad, pad)), 'constant')
    
    # Perform convolution on x_pad
    for n in range(N): # for each neuron
        for f in range(F): # for each filter/kernel
            for i in range(0, H_): # for each y activation
                for j in range(0, W_): # for each x activation
                    out[n, f, i, j] = (x_pad[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW] * w[f, :, :, :]).sum() + b[f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x_pad, w, b, conv_param)
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
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    x_pad, w, b, conv_param = cache
    N, F, H_, W_ = dout.shape
    N, C, H, W = x_pad.shape
    HH, WW = w.shape[2], w.shape[3]
    
    p = conv_param['pad']
    s = conv_param['stride']

    # initialize gradients
    dx = np.zeros((N, C, H - 2*p, W - 2*p))
    dw = np.zeros(w.shape)
    
    # create w_row matrix
    w_row = w.reshape(F, C*HH*WW)                            #[F x C*HH*WW]

    # create x_col matrix with values that each neuron is connected to
    x_col = np.zeros((C*HH*WW, H_*W_))                   #[C*HH*WW x H'*W']
    
    for index in range(N):
        out_col = dout[index].reshape(F, H_*W_)          #[F x H'*W']
        w_out = w_row.T.dot(out_col)                         #[C*HH*WW x H'*W']
        dx_cur = np.zeros((C, H, W))
        neuron = 0
        for i in range(0, H-HH+1, s):
            for j in range(0, W-WW+1, s):
                dx_cur[:,i:i+HH,j:j+WW] += w_out[:,neuron].reshape(C,HH,WW)
                x_col[:,neuron] = x_pad[index,:,i:i+HH,j:j+WW].reshape(C*HH*WW)
                neuron += 1
        dx[index] = dx_cur[:,p:-p, p:-p]
        dw += out_col.dot(x_col.T).reshape(F,C,HH,WW)

    
    db = np.sum(dout, axis=(0, 2, 3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by 

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # extract params 
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    # output volume size
    # note that the // division yields an int (while / yields a float)
    Hout = (H - HH) // stride + 1
    Wout = (W - WW) // stride + 1

    # create output volume tensor after maxpool
    out = np.zeros((N, C, Hout, Wout)) # output has same dims NCHW format as input

    # naive Loops
    for n in range(N): # for each neuron
        for i in range(Hout): # for each y activation
            for j in range(Wout): # for each x activation
                out[n, :, i, j] = np.amax(x[n, :, i*stride:i*stride+HH, j*stride:j*stride+WW], axis=(-1, -2))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Extract constants and shapes
    x, pool_param = cache
    N, C, H, W = x.shape
    HH = pool_param.get('pool_height', 2)
    WW = pool_param.get('pool_width', 2)
    stride = pool_param.get('stride', 2)

    # output volume size
    # note that the // division yields an int (while / yields a float)
    Hout = (H - HH) // stride + 1 
    Wout = (W - WW) // stride + 1

    # output volume size
    dx = np.zeros_like(x)
    
    # naive loops
    for n in range(N): # for each neuron
        for c in range(C): # for each channel
            for i in range(Hout): # for each y activation
                for j in range(Wout): # for each x activation
                    # pass gradient only through indices of max pool
                    ind = np.argmax(x[n, c, i*stride:i*stride+HH, j*stride:j*stride+WW])
                    ind1, ind2 = np.unravel_index(ind, (HH, WW))
                    dx[n, c, i*stride:i*stride+HH, j*stride:j*stride+WW][ind1, ind2] = dout[n, c, i, j]


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
