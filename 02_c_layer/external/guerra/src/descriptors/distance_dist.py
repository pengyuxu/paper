''' 
/* SHREC 2025
Marco Guerra

*/
'''

# Compute quantiles of distribution of distance in a point cloud

import numpy as np

def quantiles_of_distance( points , centroid , which_quantiles : list = None, **kwargs):
    ''' Compute quantiles of the distribution of distances between
    each point in points and the centroid

    PARAMS:
    points : np.array of shape (NPoints, 3)
    centroid : np.array of shape (3, )
    OUTPUT:
    quantiles : list - the list of quantiles
    '''

    ## parse optional arguments
    # NONE YET!

    if which_quantiles is None:

        # if not provided, take 1st 2nd and 3rd quartiles
        which_quantiles = [ 0.25 , 0.5 , 0.75 ] 

    Dists = []
    NPoints = points.shape[0]

    for i in range(NPoints):
        dist = np.linalg.norm( points[i , :].T - centroid )
        Dists.append(dist)

    Dists = np.array(Dists)

    quantiles = [ ]
    for q in which_quantiles:
        try:
            Q = np.quantile( Dists, q ) 
        except:
            Q = 0.0
        quantiles.append(Q)

    return quantiles

def centroid( points ):
    ''' Compute centroid of points

    PARAMS:
    points : np.array of shape (NPoints, 3)

    OUTPUT:
    centroid : np.array of shape (3, )
    '''

    centroid = np.mean( points , axis = 0 ).reshape((3,))

    return centroid

def distances_from_point(points, point):
    ''' Compute a vector of distances between
    each point in points and the point

    PARAMS:
    points : np.array of shape (NPoints, 3)
    point : np.array of shape (3, )
    OUTPUT:
    distances : np.array of shape (NPoints, 1)
    '''

    return np.linalg.norm( points - point , axis=1)

    