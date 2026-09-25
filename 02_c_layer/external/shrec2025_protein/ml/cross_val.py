import csv
import numpy as np

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.svm import LinearSVC, SVC

def main():
  training_data = {}
  with open('../dataset/train_set.csv', 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
      training_data[row['protein_id']] = row['class_id']

  dataset = []
  labels = []
  for protein_id, class_id in training_data.items():
    descriptor = np.fromfile(f'../dataset/train_set/{protein_id}.dat', dtype=np.float32)
    dataset.append(descriptor)
    labels.append(class_id)

  X = np.array(dataset)
  y = np.array(labels)

  print(X.shape)

  scores = cross_val_score(Pipeline([
      ('normalizer', Normalizer()),
      # ('classifier', ExtraTreesClassifier(max_features=None, class_weight="balanced", n_jobs=-1, random_state=42))
      # ('classifier', LinearSVC(class_weight="balanced", C=1024, random_state=42))
      ('classifier', SVC(class_weight="balanced", gamma=8, C=32, random_state=42))
  ]), X, y, cv=10, scoring='balanced_accuracy')

  print("Balanced Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

if __name__ == '__main__':
  main()
