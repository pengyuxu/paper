##### MARCO GUERRA 2025

# Generate the descriptors for classification of SHREC 2025
# This file is meant to be run either directly or via nohup

import sys, os
from pathlib import Path
import numpy as np


import numpy as np
from src.data_reader import DataSource, convertVTK_to_numpy

from src.descriptors.dscs_driver import compute_descriptors, compute_spherical_sectors_descs, process_spherical_sectors_descriptors
from src.descriptors.spherical_sectors import spherical_block_permutations

import csv

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# CHOOSE IF VERBOSE
verbose = False

std_out = sys.stdout
null_out = open(os.devnull, 'w')


if sys.version_info < (3, 10):
    raise RuntimeError("This project requires Python 3.10 or higher.")

print('** Generating descriptors **', flush=True)

# LOCALIZE AND READ DATA 

# Training set

input_training_set = DataSource( './data/data/train_set/', base_path='./data/data/train_set')
N_Files_Training = len(list(DataSource( './data/data/train_set/', base_path='./data/data/train_set'))) # count its length
print('Training set read - ', N_Files_Training, ' files', flush=True)

# Test set

input_test_set = DataSource( './data/data/test_set/', base_path='./data/data/test_set')
N_Files_Test = len(list(DataSource( './data/data/test_set/', base_path='./data/data/test_set'))) # count its length
print('Test set read - ', N_Files_Test, ' files', flush=True)

# CONVERT FILES TO NUMPY FOR FASTER ACCESS LATER

# Training set

print('Converting training VTK to Numpy\n')

if not verbose:
    sys.stdout = null_out

# Check that folder exists, if not create it
os.makedirs('./data/data/train_set_Numpy/', exist_ok=True) 

for j,s in enumerate(input_training_set):

    print(j+1, 'out of', N_Files_Training, flush=True)
    print(s, flush=True)

    points, triangles, potentials, norm_potentials = convertVTK_to_numpy(s)

    filename = os.path.basename(s)

    output_file = os.path.join( './data/data/train_set_Numpy/', filename )

    np.savez(output_file, points=points, triangles=triangles, potentials=potentials, norm_potentials=norm_potentials)

if not verbose:
    sys.stdout = std_out

# Test set

print('Converting test VTK to Numpy\n')

# Check that folder exists, if not create it
os.makedirs('./data/data/test_set_Numpy/', exist_ok=True) 

if not verbose:
    sys.stdout = null_out

for j,s in enumerate(input_test_set):

    print(j+1, 'out of', N_Files_Test, flush=True)
    print(s, flush=True)

    points, triangles, potentials, norm_potentials = convertVTK_to_numpy(s)

    filename = os.path.basename(s)

    output_file = os.path.join( './data/data/test_set_Numpy/', filename )

    np.savez(output_file, points=points, triangles=triangles, potentials=potentials, norm_potentials=norm_potentials)

if not verbose:
    sys.stdout = std_out

print('\nFinished conversion to Numpy \n', flush=True)


# COMPUTE LABELS
print('Computing labels \n', flush=True)
Truth = {}
labels = []

# Read the whole label list
with open('./data/data/train_set.csv', 'r') as csvfile:
    truths = csv.reader(csvfile)

    next(truths, None) # skip first row, it's a header
    for t in truths:

        Truth[t[0]] = int(t[1])

# Read it as in the training set, so they are ordered correctly
for f in DataSource('./data/data/train_set/'): # this way they have the same order as DataSource

    f = os.path.basename(f)
    filename,_ = os.path.splitext(f)
    
    labels.append(Truth[filename]) 

# convert to numpy
labels = np.array(labels).reshape((len(labels),))

# Check that folder exists, if not create it
os.makedirs('./data/labels/', exist_ok=True) 

#save
np.save('./data/labels/labels_train.npy', labels)

# COMPUTE DESCRIPTORS

print('Computing descriptors \n', flush=True)

# ALPHA DESCRIPTORS

print('Computing Alpha Persistent images descriptors \n', flush=True)

# Prominent points to keep
Num_Prominent = 150
# resolution of each axis for the persistent images
PersImPoints = 5
# set params for the run
# data base path
data_base = './data/data/'

Model = 'AlphaProminent'
params = { 'data':data_base}
params.update(  { 'Num_Prominent' : Num_Prominent , 'PersImPoints' : PersImPoints } )
params.update( { 'which_quantiles':[ .25, .5 , .75 ] } )

# Check that folder exists, if not create it
os.makedirs('./data/savedAlpha/', exist_ok=True) 

# Training set
print('   Training set', flush=True)

# data source
data_source = os.path.join(data_base, 'train_set/')

if not verbose:
    sys.stdout = null_out
data, _ = compute_descriptors(data_source, model=Model, **params)

if not verbose:
    sys.stdout = std_out

print('   Train data has shape ', data.shape)

#save
np.save( './data/savedAlpha/AlphaDescriptors_train.npy' , data )

# Test set

print('   Test set', flush=True)

# data source
data_source = os.path.join(data_base, 'test_set/')

if not verbose:
    sys.stdout = null_out

data, _ = compute_descriptors(data_source, model=Model, **params)

if not verbose:
    sys.stdout = std_out

print('   Test data has shape ', data.shape)

#save
np.save( './data/savedAlpha/AlphaDescriptors_test.npy' , data )

print('Finished Alpha Persistent images descriptors \n', flush=True)
# SPHERICAL SECTORS DESCRIPTORS

print('Spherical sectors descriptors \n', flush=True)
# Training set

# Check that folder exists, if not create it
os.makedirs('./data/sectors/train_set', exist_ok=True)
os.makedirs('./data/sectors/test_set', exist_ok=True)
os.makedirs('./data/savedSectors/', exist_ok=True)


print('   Training set \n', flush=True)

data_source = './data/data/train_set/'
numpy_source = './data/data/train_set_Numpy/'
raw_path = './data/sectors/train_set/'

print('      Compute raw')
if not verbose:
    sys.stdout = null_out
compute_spherical_sectors_descs(data_source , numpy_source, raw_path)
if not verbose:
    sys.stdout = std_out

length_of_quantiles = 3 # how many quantiles we are using
write_path = './data/savedSectors/train_set.npy'

print('      Process')
if not verbose:
    sys.stdout = null_out
process_spherical_sectors_descriptors(data_source, raw_path, numpy_source, write_path, length_of_quantiles)
if not verbose:
    sys.stdout = std_out
# Test set

print('   Test set \n', flush=True)

data_source = './data/data/test_set/'
numpy_source = './data/data/test_set_Numpy/'
raw_path = './data/sectors/test_set/'

print('      Compute raw')
if not verbose:
    sys.stdout = null_out
compute_spherical_sectors_descs(data_source , numpy_source, raw_path)
if not verbose:
    sys.stdout = std_out

length_of_quantiles = 3 # how many quantiles we are using
write_path = './data/savedSectors/test_set.npy'

print('      Process')
if not verbose:
    sys.stdout = null_out
process_spherical_sectors_descriptors(data_source, raw_path, numpy_source, write_path, length_of_quantiles)
if not verbose:
    sys.stdout = std_out
    
print('Finished spherical sectors descriptors \n', flush=True)

# DATA CLEANING: NORMALIZATION, DECORRELATION

print('Start data cleaning: normalization, decorrelation, etc \n', flush=True)

# ALPHA Descriptors

print('   Alpha descriptors')
dataTrain = np.load('./data/savedAlpha/AlphaDescriptors_train.npy')
dataTest = np.load('./data/savedAlpha/AlphaDescriptors_test.npy')

# scaling
SSc = StandardScaler()
dataTrain = SSc.fit_transform(dataTrain)
dataTest = SSc.fit_transform(dataTest)

# We observe the first 25 columns, corresponding to H_0, benefit from decorrelation
for i in range(5):
    c1 = 0 + 5*i
    c2 = 1 + 5*i
    cc = 2 + 5*i
    c4 = 3 + 5*i
    c5 = 4 + 5*i

    # training set

    # compute correlations
    reg1 = LinearRegression().fit(dataTrain[:, [cc]], dataTrain[:, c1])
    reg2 = LinearRegression().fit(dataTrain[:, [cc]], dataTrain[:, c2])
    reg4 = LinearRegression().fit(dataTrain[:, [cc]], dataTrain[:, c4])
    reg5 = LinearRegression().fit(dataTrain[:, [cc]], dataTrain[:, c5])

    # Compute residuals
    res1 = dataTrain[:, c1] - reg1.predict(dataTrain[:, [c1]])
    res2 = dataTrain[:, c2] - reg2.predict(dataTrain[:, [c2]])
    res4 = dataTrain[:, c4] - reg4.predict(dataTrain[:, [c4]])
    res5 = dataTrain[:, c5] - reg5.predict(dataTrain[:, [c5]])
    

    # substitute
    dataTrain[:,c1] = res1
    dataTrain[:,c2] = res2
    dataTrain[:,c4] = res4
    dataTrain[:,c5] = res5

    # test set

    # compute correlations
    reg1 = LinearRegression().fit(dataTest[:, [cc]], dataTest[:, c1])
    reg2 = LinearRegression().fit(dataTest[:, [cc]], dataTest[:, c2])
    reg4 = LinearRegression().fit(dataTest[:, [cc]], dataTest[:, c4])
    reg5 = LinearRegression().fit(dataTest[:, [cc]], dataTest[:, c5])

    # Compute residuals
    res1 = dataTest[:, c1] - reg1.predict(dataTest[:, [c1]])
    res2 = dataTest[:, c2] - reg2.predict(dataTest[:, [c2]])
    res4 = dataTest[:, c4] - reg4.predict(dataTest[:, [c4]])
    res5 = dataTest[:, c5] - reg5.predict(dataTest[:, [c5]])
    

    # substitute
    dataTest[:,c1] = res1
    dataTest[:,c2] = res2
    dataTest[:,c4] = res4
    dataTest[:,c5] = res5

np.save('./data/savedAlpha/AlphaDescriptors_train_clean.npy', dataTrain)
np.save('./data/savedAlpha/AlphaDescriptors_test_clean.npy', dataTest)

print('   Finished cleaning Alpha')
    

# SPHERICAL sectors descriptors

print('   Spherical descriptors')

dataTrain = np.load('./data/savedSectors/train_set.npy')
dataTest = np.load('./data/savedSectors/test_set.npy')

# H_2 data is redundant, it's always zero

exclude = [ 8 + 13*k for k in range(8) ]
dataTrain = np.delete(dataTrain, exclude, axis = 1)
dataTest = np.delete(dataTest, exclude, axis = 1)

# scaling
SSc = StandardScaler()
dataTrain = SSc.fit_transform(dataTrain)
dataTest = SSc.fit_transform(dataTest)


# decorrelate sectors
for k in range(8):
    c1 = 0 + 12*k
    c2 = 1 + 12*k
    c3 = 2 + 12*k
    q1 = 3 + 12*k
    q2 = 4 + 12*k
    q3 = 5 + 12*k

    # training set 

    # compute correlations
    regC1 = LinearRegression().fit(dataTrain[:, [c2]], dataTrain[:, c1])
    regC3 = LinearRegression().fit(dataTrain[:, [c2]], dataTrain[:, c3])

    regQ1 = LinearRegression().fit(dataTrain[:, [q2]], dataTrain[:, q1])
    regQ3 = LinearRegression().fit(dataTrain[:, [q2]], dataTrain[:, q3])

    # Compute residuals
    resC1 = dataTrain[:, c1] - regC1.predict(dataTrain[:, [c2]])
    resC3 = dataTrain[:, c3] - regC3.predict(dataTrain[:, [c2]])

    resQ1 = dataTrain[:, q1] - regQ1.predict(dataTrain[:, [q2]])
    resQ3 = dataTrain[:, q3] - regQ3.predict(dataTrain[:, [q2]])

    # substitute
    dataTrain[:,c1] = resC1
    dataTrain[:,c3] = resC3
    
    dataTrain[:,q1] = resQ1
    dataTrain[:,q3] = resQ3

    # test set 

    # compute correlations
    regC1 = LinearRegression().fit(dataTest[:, [c2]], dataTest[:, c1])
    regC3 = LinearRegression().fit(dataTest[:, [c2]], dataTest[:, c3])

    regQ1 = LinearRegression().fit(dataTest[:, [q2]], dataTest[:, q1])
    regQ3 = LinearRegression().fit(dataTest[:, [q2]], dataTest[:, q3])

    # Compute residuals
    resC1 = dataTest[:, c1] - regC1.predict(dataTest[:, [c2]])
    resC3 = dataTest[:, c3] - regC3.predict(dataTest[:, [c2]])

    resQ1 = dataTest[:, q1] - regQ1.predict(dataTest[:, [q2]])
    resQ3 = dataTest[:, q3] - regQ3.predict(dataTest[:, [q2]])

    # substitute
    dataTest[:,c1] = resC1
    dataTest[:,c3] = resC3
    
    dataTest[:,q1] = resQ1
    dataTest[:,q3] = resQ3

np.save('./data/savedSectors/train_set_clean.npy',dataTrain)
np.save('./data/savedSectors/test_set_clean.npy',dataTest)

print('   Finished cleaning sectors', flush=True)
print('Finished data cleaning \n', flush=True)

# DATA AUGMENTATION

print('Start data augmentation on spherical descriptors \n', flush=True)

dataTrain = np.load('./data/savedSectors/train_set_clean.npy')
dataTest = np.load('./data/savedSectors/test_set_clean.npy')

dataTrain_aug = np.zeros( ( dataTrain.shape[0] * 8 , dataTrain.shape[1] ) )
dataTest_aug = np.zeros( ( dataTest.shape[0] * 8 , dataTest.shape[1] ) )

for r in range(dataTrain.shape[0]):
    vec = dataTrain[r,:].reshape(-1)


    aug_vec = np.array( spherical_block_permutations(vec, int(dataTrain.shape[1]/8) ) )
    dataTrain_aug[ 8*r : 8*r + 8 , : ] = aug_vec

for r in range(dataTest.shape[0]):
    vec = dataTest[r,:].reshape(-1)


    aug_vec = np.array( spherical_block_permutations(vec, int(dataTest.shape[1]/8) ) )
    dataTest_aug[ 8*r : 8*r + 8 , : ] = aug_vec

np.save('./data/savedSectors/train_set_aug.npy', dataTrain_aug)
np.save('./data/savedSectors/test_set_aug.npy', dataTest_aug)

print('Augmentation on Alpha descriptors (dummy copy) \n', flush=True)

dataAlpha_train = np.load('./data/savedAlpha/AlphaDescriptors_train_clean.npy')
dataAlpha_test = np.load('./data/savedAlpha/AlphaDescriptors_test_clean.npy')

dataAlphaTrain_aug = np.repeat(dataAlpha_train, repeats=8, axis=0)
dataAlphaTest_aug = np.repeat(dataAlpha_test, repeats=8, axis=0)

np.save('./data/savedAlpha/AlphaDescriptors_train_aug.npy', dataAlphaTrain_aug)
np.save('./data/savedAlpha/AlphaDescriptors_test_aug.npy', dataAlphaTest_aug)

# Augmenting labels
print('Augmenting labels', flush=True)
labels = np.load('./data/labels/labels_train.npy')

labels_aug = np.repeat(labels, repeats=8, axis=0)

np.save('./data/labels/labels_train_aug.npy', labels_aug)

print('Finished data augmentation', flush=True)

# FINAL OUTPUT

print('Done')
print('The cleaned and augmented spherical descriptors are at ./data/savedSectors/train_set_aug.npy and ./data/savedSectors/test_set_aug.npy')
print('The cleaned and augmented Alpha descriptors are at ./data/savedAlpha/AlphaDescriptors_train_aug.npy and ./data/savedAlpha/AlphaDescriptors_test_aug.npy')
print('The augmented labels are at ./data/labels/labels_train_aug.npy')

print('** Finished **')



