import csv
import numpy as np
import time
import sys

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC

def main():
  # Load the training data
  train_data = {}
  with open('../dataset/train_set.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
      train_data[row['protein_id']] = row['class_id']
  train_dataset = []
  train_labels = []
  for protein_id, class_id in train_data.items():
    descriptor = np.fromfile(f'../dataset/train_set/{protein_id}.dat', dtype=np.float32)
    train_dataset.append(descriptor)
    train_labels.append(class_id)

  # Train the model
  X_train = np.array(train_dataset)
  y_train = np.array(train_labels)
  pipeline = Pipeline([
    ('normalizer', Normalizer()),
    ('classifier', SVC(class_weight="balanced", gamma=8, C=32, random_state=42))
  ])
  pipeline.fit(X_train, y_train)

  # Load and predict the test data
  diff_time = 0
  n_test_models = 0
  results = {}
  with open('../dataset/test_set.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
      protein_id = row['protein_id'].replace('.vtk', '')
      start_time = time.time()
      descriptor = np.fromfile(f'../dataset/test_set/{protein_id}.dat', dtype=np.float32)
      class_id = pipeline.predict(descriptor.reshape(1, -1))
      end_time = time.time()
      diff_time += end_time - start_time
      n_test_models += 1
      results[protein_id] = class_id[0]

  sorted_results = {k: results[k] for k in sorted(results.keys(), key=int)}
  print('protein_id,class_id')
  for protein_id, class_id in sorted_results.items():
    print(f'{protein_id},{class_id}')

  # Print the average time taken for prediction
  print(f'Average time taken for prediction: {diff_time / n_test_models} [s]', file=sys.stderr)

if __name__ == '__main__':
  main()
