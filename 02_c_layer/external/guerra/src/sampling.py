''' 
/* SHREC 2025
Marco Guerra

*/
'''


# Building samples

import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import StratifiedShuffleSplit


def build_samples(SIZE_OF_TRAIN_SET : int = 200, all_classes : bool = True, plot : bool = False):
    '''Function to build samples and subsamples

    PARAMS:
    SIZE_OF_TRAIN_SET : int = 200 How many files to subsample
    all_classes : bool = True If True, make sure that every class is represented at least TWICE. If needed, takes 
    away from the very frequent classes to add the missing ones
    plot : bool = False Whether to plot

    RETURNS:
    sampled_names : list - List of the sampled filenames
    sampled_labels : list - List of the sampled labels

    Unbalanced classes are an issue. There are many classes that are never selected in the sample.
    With all_classes, we correct it this way:
    - check which classes never appear
    - find those that appear often
    - remove some from the very frequent classes and substitute with rare ones
    - leave sample size untouched
    '''

    names = []
    labs = []
    
    with open('./data/train_set.csv', 'r') as csvfile:
        truths = csv.reader(csvfile)
    
        next(truths, None)
        for i,t in enumerate(truths):
    
            t[1] = int(t[1])
    
            names.append( t[0] )
            labs.append( t[1] )
    
    StrShSp = StratifiedShuffleSplit(n_splits=1, train_size=SIZE_OF_TRAIN_SET, random_state=None)
    
    # turn to array
    names = np.array(names, dtype = str)
    labs = np.array(labs)
    
    # array of selected names' ids 
    nameIds = [x for x,_ in StrShSp.split(names, labs)][0]
    
    #turn to list
    nameIds = nameIds.tolist()
    
    # select names
    sampled_names = [names[i] for i in nameIds]
    # and labels
    sampled_labels = [labs[i] for i in nameIds]

    if plot:
        countsS, binsS = np.histogram(sampled_labels, bins = np.arange(0,97))

    if all_classes:
    
        # Find the unique labels we have sampled
        idsS, cntsS = np.unique(sampled_labels, return_counts=True)
        
        # Identify the four most frequent classes
        # FOUR BECAUSE of the specific data we have, where 4 classes are enormous 
        top_freq = np.argsort(cntsS)[-4:]
        top_ids = idsS[top_freq]
        top_ids
        
        # choose one to take out wrt to those frequecies
        def pick_class_to_remove():
            return np.random.choice(top_ids, p=cntsS[top_freq] / np.sum(cntsS[top_freq]) )
        
        # Find unique labels in the whole dataset
        ids, cnts = np.unique(labs, return_counts=True)
        
        # Set difference
        missing_labs = set(ids.tolist()) - set(idsS.tolist())
        
        # For each missing class, remove one of the most frequent ones and replace it with one of these
        for l in missing_labs:
            
            # find two to remove
            class_to_remove1 = pick_class_to_remove()
            entry_to_remove1 = np.random.choice(np.argwhere( sampled_labels == class_to_remove1).flatten())

            class_to_remove2 = pick_class_to_remove()
            entry_to_remove2 = np.random.choice(np.argwhere( sampled_labels == class_to_remove2).flatten())

            while entry_to_remove2 == entry_to_remove1:
                entry_to_remove2 = np.random.choice(np.argwhere( sampled_labels == class_to_remove2).flatten())
   
            assert sampled_labels[entry_to_remove1] == class_to_remove1
            assert sampled_labels[entry_to_remove2] == class_to_remove2
        
            # find two to add
            
            to_add = np.random.choice(np.argwhere( labs == l).flatten(),2,replace = False)

            to_add1 = to_add[0]
            to_add2 = to_add[1]
 
        
            sampled_names[entry_to_remove1] = names[to_add1]
            sampled_labels[entry_to_remove1] = labs[to_add1]

            sampled_names[entry_to_remove2] = names[to_add2]
            sampled_labels[entry_to_remove2] = labs[to_add2]
        
            assert sampled_labels[entry_to_remove1] == l
            assert sampled_labels[entry_to_remove2] == l

        # now those that appeared only once!

        # update
        idsS, cntsS = np.unique(sampled_labels, return_counts=True)

        onces = np.argwhere(cntsS == 1).flatten().tolist()
        for l in onces:

            # find one to remove
            class_to_remove = pick_class_to_remove()
            entry_to_remove = np.random.choice(np.argwhere( sampled_labels == class_to_remove).flatten())

            # find one to add
            to_add = np.random.choice(np.argwhere( labs == l).flatten())

            # check it is new!
            while names[to_add] in sampled_names:
                to_add = np.random.choice(np.argwhere( labs == l).flatten())

            sampled_names[entry_to_remove] = names[to_add]
            sampled_labels[entry_to_remove] = labs[to_add]
    
    if plot:

        counts, bins = np.histogram(labs, bins = np.arange(0,97))
        countsSC, binsSC = np.histogram(sampled_labels, bins = np.arange(0,97))
        
        if all_classes:
            fig, ax = plt.subplots(3, 1, figsize=(6, 10))
        else:
            fig, ax = plt.subplots(2, 1, figsize=(6, 10))
        
        ax[0].bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', alpha=0.7)
        ax[0].set_xlabel('Class')
        ax[0].set_ylabel('Frequency')
        ax[0].set_title('Full dataset')
        
        
        ax[1].bar(binsS[:-1], countsS, width=np.diff(bins), edgecolor='black', alpha=0.7)
        ax[1].set_xlabel('Class')
        ax[1].set_ylabel('Frequency')
        ax[1].set_title('Sampled dataset')

        if all_classes:
            ax[2].bar(binsSC[:-1], countsSC, width=np.diff(bins), edgecolor='black', alpha=0.7)
            ax[2].set_xlabel('Class')
            ax[2].set_ylabel('Frequency')
            ax[2].set_title('Corrected sampled dataset')

        plt.tight_layout()

        return sampled_names, sampled_labels, ax


    return sampled_names, sampled_labels