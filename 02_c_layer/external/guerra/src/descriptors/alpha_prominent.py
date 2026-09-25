''' 
/* SHREC 2025
Marco Guerra

*/
'''

# Descriptors for data

from pathlib import Path

import numpy as np
from gudhi import AlphaComplex
from gudhi.representations import ProminentPoints, PersistenceImage

def AlphaDiag(points, N_Prominent : int | float = 100):
    '''Compute Alpha persistence diagram, and return only the 
        prominent points. Keeps those far from the diagonal.

    PARAMS:
    points: input points
    N_Prominent : int  = 100 - Number of prominent points to keep.
    RETURNS:
    Dgm0, Dgm1, Dgm2 - The diagrams in dimension 0,1,2
    
    '''
    
    st = AlphaComplex(points=points).create_simplex_tree()
    st.compute_persistence(homology_coeff_field=2, persistence_dim_max=2)
    diag = st.persistence()

    # Turn to numpy
    dgm0 = np.array( [ x[1] for x in diag if x[0] == 0 if x[1][1] != np.inf ] )
    dgm1 = np.array( [ x[1] for x in diag if x[0] == 1 ] )
    dgm2 = np.array( [ x[1] for x in diag if x[0] == 2 ] )
    
    # Prominent points layer
    PrP = ProminentPoints(use = True, num_pts = N_Prominent, location='upper')
    
    Dgm0 = PrP(dgm0)
    Dgm1 = PrP(dgm1)
    Dgm2 = PrP(dgm2)
    
    # Pad with zeros if necessary
    for dgm in [Dgm0, Dgm1, Dgm2]:
        missing = N_Prominent - dgm.shape[0]
        if missing > 0:
            dgm = np.pad( dgm, ((0,missing), (0,0)) )

    return Dgm0, Dgm1, Dgm2

def PersImagesVectorize( Dgm0, Dgm1, Dgm2 , res : int = 10 ):
    # Vectorize with persistence images
    PerIm = PersistenceImage(resolution=(res,res))
    Img0 = PerIm(Dgm0)
    Img1 = PerIm(Dgm1)
    Img2 = PerIm(Dgm2)
    
    Img = np.concatenate((Img0, Img1, Img2), axis=0, dtype = float)

    return Img