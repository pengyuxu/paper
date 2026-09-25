''' Marco Guerra - IMATI 2025

SHREC25 challenge.
'''

# Data reader file to read vtk files

import sys, os
import numpy as np
from pathlib import Path

def filesIn(path):
    '''List all non-hidden files in path. Exclude directories and filnames starting with .
       Sort alphabetically
    '''
    
    for file in os.listdir(path):
        if not file.startswith('.') and os.path.isfile(os.path.join(path, file)):
            
            yield file

class DataSource:
    ''' Class to read data, either a folder or a list of filenames

    PARAMS:
    source : str | Path | list - if list, assumes it is a list of filenames to read. If they are absolute
    paths, base_path should be None and they are simply returned as an iterator. If not, base_path is prefixed
    to them. If a str or a Path, it is assumed that os.path.isdir(source) will return True, and the return 
    value is an iterator of the regular, non-hidden files in the directory. In all cases, the files are 
    returned in alphabetical order.
    
    base_path : str | Path | None. Defaults to None. Otherwise, it is assumed that it is a path and that
    os.path.join(base_path, x) is valid for each element of source.
    '''

    def __init__( self, source : str | Path | list , base_path : str | Path = None):
        '''source can be a list of strings, or a Path or str
        '''

        if isinstance( source, list ): # if it's a list of filenames

            self.FilesList = source
            self.index = 0
            self.high = len(source)
            
            if base_path is not None:
                self.FilesList =  [os.path.join(base_path, f) for f in self.FilesList ]

            self.FilesList = [os.path.abspath(f) for f in self.FilesList]

        elif os.path.isdir( source ): # if it is a folder

            self.FilesList = list(sorted(filesIn( source)))
            self.FilesList = [ os.path.abspath( os.path.join(source, f) ) for f in self.FilesList ]
            self.index = 0
            self.high = len(self.FilesList)
            self.base_path = None

        else:
            raise ValueError("Input must be a valid folder path or a list of valid filenames.")

    def __iter__(self):
        return self

    def __next__(self):
        
        if self.index >= len(self.FilesList):
            raise StopIteration

        file = self.FilesList[self.index]
        self.index += 1


        return Path(file)

def read_vertices_VTK( source_file : str | Path , out_file : str | Path = None , out_var : bool = None):
    '''Read a VTK source file and parse the vertices. It is assumed the vertices appear as the
    first field in the file. Store the output either to a file out_file or return it as a 
    numpy array.

    PARAMS:
    source_file : str | Path. An input vtk file from which to read the points
    out_file : str | Path. An output file to write the point_cloud to
    out_var : bool. Whether to return the points as a np array
    '''

    with open(source_file, 'r') as f:
        
        for _ in range(5): # The first five lines are a constant header
            f.readline()

        # the fifth line contains the number of points as the second word
        line = f.readline().strip().split()
        N_Verts = int(line[1])

        if out_file is not None:
            try:
                write_file = open(out_file, 'w')
            except IOError:
                print('Output file '+ out_file + ' cannot be opened')
                return 

            def add_point( p ):

                writeline = ' '.join(str(x) for x in p) + '\n'
                write_file.write( writeline )

        elif out_var:

            matrix = np.zeros( ( N_Verts, 3 ) )
            def add_point( p ):

                matrix[ line_counter - 1 , : ] = p

        else:
            raise ValueError('One of out_file or out_var must be set')

        line_counter = 1

        for line in f: 
        
            if line_counter >= N_Verts:
                break

            points = line.strip().split()
            points = [float(x) for x in points]

            add_point( points )
            
            line_counter += 1

        if out_var:
            return matrix

def num_vertices_VTK( source_file : str | Path):
    '''Read a VTK source file and parse the header to read how many vertices it has. 

    PARAMS:
    source_file : str | Path. An input vtk file from which to read the points

    RETURNS:
    int: number of vertices
    '''

    with open(source_file, 'r') as f:
        
        for _ in range(5): # The first five lines are a constant header
            f.readline()

        # the fifth line contains the number of points as the second word
        line = f.readline().strip().split()
        N_Verts = int(line[1])

    return N_Verts

def convertVTK_to_numpy( source_file : str | Path ):
    '''Read a VTK source file and parse the contents. First the vertices, then 
    the triangles, next the potentials, finally the normal potentials.
    It is assumed the fields appear in this order.
    Store the output and return it as numpy arrays.

    PARAMS:
    source_file : str | Path. An input vtk file from which to read the points

    OUTPUT:
    vertices : np.array of shape (NVerts, 3), dtype float
    triangles : np.array of shape (NTris, 3), dtype int
    potentials : np.array of shape (NVerts, 3), dtype float
    normal_potentials : np.array of shape (NVerts, 3), dtype float
    '''

    with open(source_file, 'r') as f:
        
        for _ in range(5): # The first five lines are a constant header
            f.readline()

        # the fifth line contains the number of points as the second word
        line = f.readline().strip().split()
        N_Verts = int(line[1])

        points = np.zeros( (N_Verts , 3), dtype=float )

        line_counter = 0
        
        for line in f: 
        
            if line_counter >= N_Verts:
                break

            NewP = line.strip().split()
            NewP = [float(x) for x in NewP]

            points[ line_counter , : ] = np.array(NewP)
            
            line_counter += 1

        # the next line contains the number of TRIANGLES as the second/third word??
        # line = f.readline().strip().split() ->> it's already READ!!
        line = line.strip().split()
        N_Tris = int(line[1])

        triangles = np.zeros( (N_Tris , 3), dtype=int )

        line_counter = 0

        for line in f: 
        
            if line_counter >= N_Tris:
                break

            NewT = line.strip().split()
            NewT = [int(x) for x in NewT]

            # first entry is the number 3
            NewT = NewT[1:]

            triangles[ line_counter , : ] = np.array(NewT)
            
            line_counter += 1

        # the next 3 lines contains the Potentials header
        line = f.readline() # we already know the length is N_Verts
        line = f.readline()
        line = f.readline()

        potentials = np.zeros( (N_Verts , 1), dtype=float )

        line_counter = 0

        for line in f: 
        
            if line_counter >= N_Verts:
                break

            NewP = line.strip().split()
            NewP = [float(x) for x in NewP]

            potentials[ line_counter , : ] = NewP[0]
            
            line_counter += 1

        # the next line contains the Normal Potentials header
        line = f.readline().strip().split() # we already know the length is N_Verts

        norm_potentials = np.zeros( (N_Verts , 1), dtype=float )

        line_counter = 0

        for line in f: 
        
            if line_counter >= N_Verts:
                break

            NewNP = line.strip().split()
            NewNP = [float(x) for x in NewNP]

            norm_potentials[ line_counter , : ] = NewNP[0]
            
            line_counter += 1

    return points, triangles, potentials, norm_potentials

        
    

        
        

            