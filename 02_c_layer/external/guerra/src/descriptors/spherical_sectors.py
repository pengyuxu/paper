''' 
/* SHREC 2025
Marco Guerra

*/
'''

import numpy as np

# Computations of spherical sectors from a protein point cloud and a centroid

def which_sector( p , centroid ):
    '''Find which of the 8 sectors a point belongs to, wrt the centroid

    '''
    vect = p - centroid

    if vect[2] >= 0.0: # z >= 0.0
        if vect[0] >= 0.0: # x >= 0.0

            if vect[1] >= 0.0: # y >= 0.0
                return 1
            else: # y < 0.0
                return 4
        else: # x < 0.0
            if vect[1] >= 0.0: # y >= 0.0
                return 2
            else: # y < 0.0
                return 3
        
    else: # z < 0.0
        if vect[0] >= 0.0: # x >= 0.0

            if vect[1] >= 0.0: # y >= 0.0
                return 5
            else: # y < 0.0
                return 8
        else: # x < 0.0
            if vect[1] >= 0.0: # y >= 0.0
                return 6
            else: # y < 0.0
                return 7

def common_sector( simplex, centroid ):
    '''Find which common sector a list of vertices belongs to. Raise
    ValueError if it does not exist

    '''
    sectors = np.array([which_sector(p, centroid) for p in simplex], dtype = int)
    unique_sec = np.unique(sectors)

    if unique_sec.shape[0] > 1:
        raise ValueError

    else:
        return unique_sec[0]

def spherical_permutations( rows , columns, size ):
    ''' Generate the permutations corresponding to rotations on the sphere.

    data is assumed to be a vector of shape rows * columns * size. Its blocks
    of index 1 through 8 are assumed to be spatially placed like
    1 2 3 4
    5 6 7 8
    The permutations are the cyclic ones in rows and columns. They are 8, one
    of which is the identity.
    '''

    def switch_rows():
        half = int(rows*columns*size / 2 )
        ids = list( range( half, rows*columns*size )) + list(range(half))
        return np.array(ids)

    def cycle_columns():
        ids = np.arange(rows*columns*size, dtype=int)
        swap = np.arange(1,size)
        swap = np.concatenate([swap , np.zeros( (1), dtype=int )])
        
        for b in range(rows * columns):
            ids[ b*size : (b+1)*size ] = ids[ b*size : (b+1)*size ][swap]
        return ids

    perms = []
    perm = np.arange(rows*columns*size)
    for c in range(columns):
        perms.append(perm)
        perm = perm[cycle_columns()]

    return perms

def spherical_block_permutations(vec, B):
    """
    Given a flattened 2x4 matrix of blocks (each block has B entries),
    return all 8 permutations under cyclic column shifts and row swaps.
    
    Parameters:
        vec (list or np.ndarray): Flattened array of length 8*B.
        B (int): Size of each block (number of entries per matrix cell).
        
    Returns:
        List of np.ndarray: 8 permutations, each of shape (8*B,).
    """
    vec = np.asarray(vec)
    assert vec.size == 8 * B, "Input must be a flattened 2x4 matrix of blocks."
    
    # Reshape into (2 rows, 4 columns, B elements per block)
    mat = vec.reshape((2, 4, B))
    
    permutations = []
    for shift in range(4):
        # Shift columns by `shift`
        shifted = np.roll(mat, shift=-shift, axis=1)
        
        # Flatten to 1D
        permutations.append(shifted.reshape(-1))
        
        # Add the row-flipped version
        flipped = shifted[::-1, :, :]  # swap rows
        permutations.append(flipped.reshape(-1))
    
    return permutations