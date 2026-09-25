''' 
/* SHREC 2025
Marco Guerra

*/
'''

# Descriptors for data

import sys, os

from pathlib import Path

import numpy as np
from gudhi import AlphaComplex, SimplexTree
from gudhi.representations import ProminentPoints, PersistenceImage

from src.data_reader import DataSource, read_vertices_VTK, num_vertices_VTK
from .alpha_prominent import AlphaDiag, PersImagesVectorize
from .distance_dist import quantiles_of_distance, distances_from_point
from .distance_dist import centroid as Centroid
from .spherical_sectors import common_sector, which_sector

import csv
import pickle

def compute_descriptors( data_source : str | Path, model : str, **kwargs ):
    '''Helper function to compute descriptors choosing the appropriate 
    combination of methods from below

    '''

    ## parse optional arguments
    Num_Prominent = kwargs.get('Num_Prominent', None)  # Default to None if not provided
    PersImPoints = kwargs.get('PersImPoints', None)
    which_quantiles = kwargs.get('which_quantiles', None)

    source = DataSource( data_source, base_path=data_source)
    N_Files = len(list(source))
    
    # vector of labels
    labels = []

    if model == 'AlphaProminent':
        data = np.zeros( (N_Files , 3*(PersImPoints**2)) )

    if model == 'quantiles':
        data = np.zeros( ( N_Files , len(which_quantiles)) )

    if model == 'Combined':

        # 3 quantiles, 3 cumulative radial potentials, num of pairs with lifetime > threshold
        # for dim 0,1,2 , birth and death of longest H_0 bar, potential at corresponding 
        # generating vertices
        N_Features = len(which_quantiles) + len(which_quantiles) + 3 + 2 + 2
        data = np.zeros(( N_Files ,N_Features) , dtype = float)

    if model == 'Sectors':

        N_Features = len(which_quantiles) + len(which_quantiles) + 2 + 2 + 2 # NO H2!!
        data = np.zeros(( N_Files ,8*N_Features) , dtype = float) # For 8 sectors!!
    
    # # read the labels
    # Truth = {}
    # with open('./data/data/train_set.csv', 'r') as csvfile:
    #     truths = csv.reader(csvfile)
    
    #     next(truths, None) # skip first row, it's a header
    #     for t in truths:

    #         Truth[t[0]] = int(t[1])
    
    source = DataSource( data_source, base_path=data_source)

    for j,s in enumerate(source):

        print(j+1, 'out of', N_Files, flush=True)
        print(s, flush=True)

        # Find the label for the protein we are reading
        filename = os.path.basename(s)
        filename = os.path.splitext(filename)[0]
        #labels.append(Truth[filename])


        if model == 'AlphaProminent':

            points = read_vertices_VTK(s , out_var=True).tolist()
            
            Dgm0, Dgm1, Dgm2 = AlphaDiag(points, N_Prominent = Num_Prominent)
            Img = PersImagesVectorize(Dgm0, Dgm1, Dgm2, res = PersImPoints)
            data[j,:] = Img

        if model == 'quantiles':

            points = read_vertices_VTK(s , out_var=True)

            center = centroid(points)

            quantiles = quantiles_of_distance(points, center, which_quantiles)
            data[j,:] = np.array(quantiles).reshape( (1, len(which_quantiles)) )

        if model == 'Combined':

            
            row = np.zeros( ( N_Features ) , dtype = float)

            read_data_file = './data/data/train_set_Numpy/' + filename + '.vtk.npz'
            res = np.load( read_data_file, allow_pickle=False )

            points = res['points']
            center = centroid(points)

            N_Verts = points.shape[0]

            quantiles = quantiles_of_distance(points, center, which_quantiles)
            
            row[0:len(which_quantiles)] = np.array(quantiles).reshape( (1, len(which_quantiles)) )

            dists = distances_from_point(points, center) # vector of distances of each point from centroid

            ordering = np.argsort(dists) # sort them by distance
            
            potentials = res['potentials']

            radial_charge = np.cumsum( potentials[ordering] ) # potentials ordered by closest to farthest from centroid, cumulative sum

            significant_entries = [ np.floor( x * N_Verts ) for x in which_quantiles] # entries of radial charge corresponding to the desired quantiles
            significant_entries = np.array(significant_entries, dtype = int)

            cumulative_charge_at_quantiles = radial_charge[ significant_entries ] # pick the corresponding charges

            row[len(which_quantiles):2*len(which_quantiles)] = cumulative_charge_at_quantiles

            read_pers_file = './data/data/sublevelset_filtrations/train_set/' + filename + '.vtk.npz'

            res = np.load( read_pers_file, allow_pickle=True )

            dgm0 = res['dgm0']
            dgm1 = res['dgm1']
            dgm2 = res['dgm2']
            gens = res['gens']

            Lambda = 1.0/10
            Threshold = quantiles[1] * Lambda # TRY ONE TENTH OF THE MEDIAN

            pers0 = dgm0[:,1] - dgm0[:,0]
            pers1 = dgm1[:,1] - dgm1[:,0]
            pers2 = dgm2[:,1] - dgm2[:,0]

            pers0 = dgm0[:,1] - dgm0[:,0]
            mask0 = (pers0 >= Threshold) & (pers0 < np.inf)
            count0 = np.sum(mask0)

            pers1 = dgm1[:,1] - dgm1[:,0]
            mask1 = (pers1 >= Threshold) & (pers1 < np.inf)
            count1 = np.sum(mask1)

            pers2 = dgm2[:,1] - dgm2[:,0]
            mask2 = (pers2 >= Threshold) & (pers2 < np.inf)
            count2 = np.sum(mask2)

            row[2*len(which_quantiles):2*len(which_quantiles)+3] = np.array([ count0 , count1 , count2 ])

            longest0_ind = np.argmax(pers0[ pers0 < np.inf])
            
            longest0 = dgm0[longest0_ind,:] # longest interval in H_0

            row[2*len(which_quantiles)+3:2*len(which_quantiles)+5] = longest0

            # Find generating vertices and their potentials
            gen_verts = gens[0][0][longest0_ind,:] # gens[0] is finite pairs, [0] is dimension 0, next index is which entry

            gen_potentials = potentials[ gen_verts ].reshape((2,)) # Find the potentials at the vertices creating the largest CC

            row[2*len(which_quantiles)+5:] = gen_potentials

            data[j,:] = row
            
            

    return data, labels

def compute_spherical_sectors_descs(data_source : str | Path, NumpySource : str | Path, out_path : str | Path ,  **kwargs):
    ''' Compute the required descriptors on the spherical sectors rather
    than on the whole protein.

    For each protein file, save a file containing a dictionary with 8 fields.
    Each field is one sector, containing a list of the raw descriptors:
    [ quantiles , radial_accumulated_potential , Dgm0, Dgm1, gens ]
    (gens is the generators of each birth and death event).

    Assumes data have been cast to a numpy serialized file.
    Outputs on out_path.

    '''

    source = DataSource( data_source, base_path=data_source)
    N_Files = len(list(source))
    
    source = DataSource( data_source, base_path=data_source)
    
    which_quantiles = kwargs.get('which_quantiles', [.25 , .5 , .75])
    
    for j,s in enumerate(source):
    
        print(j+1, 'out of', N_Files, flush=True)
        print(s, flush=True)
    
        filename = os.path.basename(s)
        filename, _ = os.path.splitext(filename)
    
        # read points, triangles, potentials, etc
    
        read_data_file = NumpySource + filename + '.vtk.npz'
        res = np.load( read_data_file, allow_pickle=False )
    
        points = res['points']
        triangles = res['triangles']
        potentials = res['potentials']
    
        N_Verts = points.shape[0]
    
        centroid = Centroid(points)
        dists = distances_from_point(points, centroid)
    
        ## NOW SPLIT IN 8 SECTORS
    
        SecPoints = []
        SCs = []
        for _ in range(8):
            SCs.append( SimplexTree() )
            SecPoints.append( [] )
    
        for i in range(triangles.shape[0]):
    
            t = triangles[i,:]
            a,b,c = t[0] , t[1] , t[2] 
            
            tri = [ a,b,c ] 
    
            try:
                sector = common_sector([points[x,:] for x in tri ], centroid)
    
            except ValueError:
                continue
            
            #print(tri)
            SCs[sector-1].insert( tri )
            SCs[sector-1].assign_filtration(tri , filtration=np.max( dists[[a,b,c]] ))
        
            e1 = [a,b]
            e2 = [a,c]
            e3 = [b,c]
        
            SCs[sector-1].insert( e1 )
            SCs[sector-1].assign_filtration(e1 , filtration=np.max( dists[[a,b]] ))
        
            SCs[sector-1].insert( e2 )
            SCs[sector-1].assign_filtration(e2 , filtration=np.max( dists[[a,c]] ))
        
            SCs[sector-1].insert( e3 )
            SCs[sector-1].assign_filtration(e3 , filtration=np.max( dists[[b,c]] ))
            
        for p in range(points.shape[0]):
    
            sector = which_sector(points[p,:] , centroid)
    
            SecPoints[sector-1].append( p )
        
            SCs[sector-1].insert([p])
            SCs[sector-1].assign_filtration([p] , filtration = dists[p] )
    
    
        res = {}
        
        for i,st in enumerate(SCs):
            st.compute_persistence()

            st.set_dimension(3)
            
            dgm0 = st.persistence_intervals_in_dimension(0)
            dgm1 = st.persistence_intervals_in_dimension(1)
            dgm2 = st.persistence_intervals_in_dimension(2)

            # this is wrt the local indexing, but we must get it wrt the global one
            gens = st.lower_star_persistence_generators()
                    
    
            if st.num_vertices() > 0:
    
                # points in this sector
                thisPoints = points[SecPoints[i], :]
                quantiles = quantiles_of_distance(thisPoints, centroid, which_quantiles)
        
                ordering = np.argsort(dists[SecPoints[i]]) # sort them by distance
        
                SecP = np.array(SecPoints[i], dtype = int)
        
                radial_charge = np.cumsum( potentials[SecP[ordering]] ) # potentials ordered by closest to farthest from centroid, cumulative sum
        
                significant_entries = [ np.floor( x * thisPoints.shape[0] ) for x in which_quantiles] # entries of radial charge corresponding to the desired quantiles
                significant_entries = np.array(significant_entries, dtype = int)
        
                cumulative_charge_at_quantiles = radial_charge[ significant_entries ] 
            else: 
                quantiles = [ 0.0 ] * len(which_quantiles)
                cumulative_charge_at_quantiles = np.zeros( ( len(which_quantiles) ) )
    
            res[i+1] = [ quantiles, cumulative_charge_at_quantiles, dgm0 , dgm1, dgm2, gens ]
    
    
        with open(os.path.join( out_path, filename ) , 'wb') as out_file:
            pickle.dump(res , out_file)



def process_spherical_sectors_descriptors(data_source : str | Path, raw_descriptors : str | Path, numpy_source : str | Path, 
                                          out_path : str | Path, len_quant : int):
    '''This function takes the files saved by compute_spherical_sectors_descs and returns a 
    numpy matrix containing the processed dataset. 
    The required steps are: choosing the dgm information and the generators. 
    '''

    source = DataSource( data_source, base_path=data_source)
    N_Files = len(list(source))
    
    source = DataSource( data_source, base_path=data_source)

    N_Features = len_quant*2 + 3 + 2 + 2
    N_sectors = 8
    data = np.zeros( (N_Files , N_sectors* N_Features) )

    for j,s in enumerate(source):
    
        print(j+1, 'out of', N_Files, flush=True)
        print(s, flush=True)

        #row = np.array()

        filename = os.path.basename(s)
        filename, _ = os.path.splitext(filename)

        # unfortunately we need the larger potentials vector
        read_data_file = numpy_source + filename + '.vtk.npz'
        res = np.load( read_data_file, allow_pickle=False )

        Potentials = res['potentials']
    
        # read descriptors
    
        read_data_file = raw_descriptors + filename
        read_dict = np.load( read_data_file, allow_pickle=True )

        row = np.array([])

        sectors = list(read_dict.keys())

        for sect in sectors:

            res = read_dict[sect]

            quantiles = np.array(res[0])
            potentials = np.array(res[1])
            dgm0 = res[2]
            dgm1 = res[3]
            dgm2 = res[4]
            gens = res[5]
    
            Lambda = 1.0/10
            Threshold = quantiles[1] * Lambda # TRY ONE TENTH OF THE MEDIAN
    
            pers0 = dgm0[:,1] - dgm0[:,0]
            mask0 = (pers0 >= Threshold) & (pers0 < np.inf)
            count0 = np.sum(mask0).reshape( (1) )
    
            pers1 = dgm1[:,1] - dgm1[:,0]
            mask1 = (pers1 >= Threshold) & (pers1 < np.inf)
            count1 = np.sum(mask1).reshape( (1) )

            # There is no death to H_2 as we introduce no tetrahedra!
            pers2 = dgm2[:,1] - dgm2[:,0]
            mask2 = (pers2 >= Threshold) # so no condition on death time!
            count2 = np.sum(mask2).reshape( (1) )

            try: # if empty use 0.0
                longest0_ind = np.argmax(pers0[ pers0 < np.inf])
            except ValueError:
                longest0 = np.array([ 0.0 , 0.0 ])
                gen_potentials = np.zeros( (2) , dtype=float )
            else:
                longest0 = dgm0[longest0_ind,:] # longest interval in H_0
                gen_verts = gens[0][0][longest0_ind,:] # gens[0] is finite pairs, [0] is dimension 0, next index is which entry
                gen_potentials = Potentials[ gen_verts ].reshape((2,)) # Find the potentials at the vertices creating the largest CC


            sect_row = np.concatenate( [quantiles , potentials , count0 , count1 , count2 , longest0 , gen_potentials] )

            row = np.concatenate( [row , sect_row] )


        data[ j , : ] = row


    np.save(out_path, data)

        
    
    
    
        

















