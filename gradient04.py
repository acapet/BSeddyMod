# -*- coding: utf-8 -*-

# %run gradient04.py

# https://github.com/Unidata/MetPy/issues/174
# https://gist.github.com/deeplycloudy/1b9fa46d5290314d9be02a5156b48741


from numpy import zeros
#from numexpr import evaluate
#from numba import jit

#@jit
def gradient(f, *varargs):
    """Calculate the fourth-order-accurate gradient of an N-dimensional scalar function.
    Uses central differences on the interior and first differences on boundaries
    to give the same shape.
    Inputs:
      f -- An N-dimensional array giving samples of a scalar function
      varargs -- 0, 1, or N scalars giving the sample distances in each direction
    Outputs:
      N arrays of the same shape as f giving the derivative of f with respect
       to each dimension.
    """
    N = len(f.shape)  # number of dimensions
    n = len(varargs)
    if n == 0:
        dx = [1.0]*N
    elif n == 1:
        dx = [varargs[0]]*N
    elif n == N:
        dx = list(varargs)
    else:
        #raise SyntaxError, ("invalid number of arguments")
        raise SyntaxError(("invalid number of arguments"))
    # use central differences on interior and first differences on endpoints

    #print dx
    outvals = []

    # create slice objects --- initially all are [:, :, ..., :]
    slice0 = [slice(None)]*N
    slice1 = [slice(None)]*N
    slice2 = [slice(None)]*N
    slice3 = [slice(None)]*N
    slice4 = [slice(None)]*N

    otype = f.dtype.char
    if otype not in ['f', 'd', 'F', 'D']:
        otype = 'd'

    for axis in range(N):       
        # select out appropriate parts for this dimension
        out = zeros(f.shape, f.dtype.char)
        
        slice0[axis] = slice(2, -2)
        slice1[axis] = slice(None, -4)
        slice2[axis] = slice(1, -3)
        slice3[axis] = slice(3, -1)
        slice4[axis] = slice(4, None)
        
        f1, f2, f3, f4 = (f[tuple(slice1)],
                          f[tuple(slice2)],
                          f[tuple(slice3)],
                          f[tuple(slice4)])
        #out[tuple(slice0)] = evaluate("(f1 - 8.0 * f2 + 8.0 * f3 - f4) / 12.0")
        out[tuple(slice0)] = (f1 - 8.0 * f2 + 8.0 * f3 - f4) / 12.0
        
        slice0[axis] = slice(None, 2)
        slice1[axis] = slice(1, 3)
        slice2[axis] = slice(None, 2)
        # 1D equivalent -- out[0:2] = (f[1:3] - f[0:2])
        out[tuple(slice0)] = (f[tuple(slice1)] - f[tuple(slice2)])
        
        slice0[axis] = slice(-2, None)
        slice1[axis] = slice(-2, None)
        slice2[axis] = slice(-3, -1)
        ## 1D equivalent -- out[-2:] = (f[-2:] - f[-3:-1])
        out[tuple(slice0)] = (f[tuple(slice1)] - f[tuple(slice2)])

        
        # divide by step size
        outvals.append(out / dx[axis])

        # reset the slice object in this dimension to ":"
        slice0[axis] = slice(None)
        slice1[axis] = slice(None)
        slice2[axis] = slice(None)
        slice3[axis] = slice(None)
        slice4[axis] = slice(None)

    if N == 1:
        return outvals[0]
    else:
        return outvals



if __name__ == '__main__':
    
    x, y = np.meshgrid(np.arange(300.), np.arange(400.))
    ii, jj = gradient(x + y, x, y)
    

