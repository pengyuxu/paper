## Marco Guerra - Method 5 V1 - Topological descriptors

This code relates to the [SHREC paper 2025](https://doi.org/10.1016/j.cag.2025.104394)

To run this:

    . Navigate to a suitable folder
    . Clone the repository in folder .
    . Run pip install -r requirements.txt

The **necessary folder structure** within ``./`` is:

    . A folder ./data/
    . A folder ./data/data/train_set/ - This must contain (a subset of) the training set.
    . A folder ./data/data/test_set/ - This must contain (a subset of) the test set.
    . A csv file ./data/train_set.csv containing the labels of the training set (as published by the organizers)
    . A csv file ./data/test_set_ground_truth.csv containing the ground truth of the test set (as published by the organizers)

**Running the code:** 

In order to run the code, make sure the above is satisfied. Then run ``python DescriptorsGeneration.py``. This processes the files in ``./data/data/train_set/`` and ``./data/data/test_set/`` and generates the topological descriptors discussed in the paper. <br>
It creates additional folder structure and several files that are used as intermediate steps in the process. Additionally, it processes the file ``./data/train_set.csv`` to obtain the training labels; it then perform some (hard-coded) feature engineering on the descriptors, such as scaling and decorrelating (some of) the features. This step is based on experience, changing the descriptors will require different choices at this step. Next, it performs data augmentation, both on training and test data, by using the specific rotation group described in the paper. Everything is saved as ``.npy`` files. <br><br>

The next step requires running ``python Learning.py``. This sets up the learning pipeline. The ground truth for the test set is read from ``./data/ground_truth_test_set.csv``, the neural network is set up and training is performed on 5000 epochs, with early stopping. The model is saved to ``./data/trained_models/`` so it can be recalled later. After training is complete, the model is run on the test set, so 8 predictions are obtained for each protein to test. A majority vote procedure ensues, to obtain the final prediction, which is saved to ``./data/prediction.npy``.

**Dummy datasets:**

In order to verify that everything works fine, it is advisable to first run the code on a very small subset of the training and test sets. This can be achieved by coping only a few input files to the relevant folders. This ensures the code is run within minutes, instead of hours. However, this has **one very important caveat:** since there are 97 protein classes, which are unlikely to be all represented by a very small subset of the data, when using dummy datasets one must **uncomment lines 66 and 67 in Learning.py**. These lines replace the true labels and ground truths with dummy binary ones, ensuring the cross-entropy loss does not fail. Obviously this entails that the results obtained are pure noise, but the purpose of this step is just to verify everything works. 

**Verbosity:**

At line 23 in ``DescriptorsGeneration.py``, the user can choose if they want the output to be very verbose and detailed, or less so. Use accordingly. 

**Remote execution:**

When using the full dataset, it is to be expected that the generation of the descriptors will take several hours. In this case, the code can be run remotely, independently of a shell, by running something like ``nohup python DescriptorsGeneration.py > nohup.out &`` . This way the shell can be exited and the result recovered at a later time. 