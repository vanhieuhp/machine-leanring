from sklearn.datasets import load_iris

import numpy as np

from entropy.scratch import build_tree, predict, to_features_dict

iris = load_iris()
X, y = iris.data, iris.target

print(X.shape)
print(y[:100])
print(iris.feature_names)
print(iris.target_names)

def discretize_with_threshold(X, feature_names, thresholds=None):
    if thresholds is None:
        thresholds = {}
        for i, name in enumerate(feature_names):
            thresholds[name] = {np.median(X[:,i])}

    result = []
    for i in range(X.shape[0]):
        sample = {}
        for j, name in enumerate(feature_names):
            if X[i,j] >= thresholds[name]:
                sample[name] = 'cao'
            else:
                sample[name] = 'thap'
        result.append(sample)

    return result, thresholds

def discretize(X, feature_name):
    result = []
    for i in range(X.shape[0]):
        sample = {}
        for j, name in enumerate(feature_name):
            median = np.median(X[:, j])
            if X[i, j] >= median:
                sample[name] = 'cao'
            else:
                sample[name] = 'thap'
        result.append(sample)
    return result

labels = [iris.target_names[i] for i in y]

# train
samples = discretize(X, iris.feature_names)
features_dict = to_features_dict(samples)
tree = build_tree(features_dict, labels)

# predict
correct = 0
for sample, label in zip(samples, labels):
    pred = predict(tree, sample)
    if pred == label:
        correct += 1

accuracy = correct / len(labels)
print(f"Accuracy: {accuracy:.2%}")

# Shuffle data trước
indices = list(range(len(samples)))
import random
random.shuffle(indices)

# 80% train, 20% test
split = int(0.8 * len(samples))
train_idx = indices[:split]
test_idx  = indices[split:]

train_samples = [samples[i] for i in train_idx]
train_labels  = [labels[i] for i in train_idx]

test_samples = [samples[i] for i in test_idx]
test_labels  = [labels[i] for i in test_idx]

# Train trên train set, test trên test set
train_features = to_features_dict(train_samples)
test_features  = to_features_dict(test_samples)

tree = build_tree(train_features, train_labels)

correct = 0
for sample, label in zip(test_samples, test_labels):
    pred = predict(tree, sample)
    if pred == label:
        correct += 1
print(f"Accuracy: {correct/len(test_labels):.2%}")

correct_train = 0
for sample, label in zip(train_samples, train_labels):
    pred = predict(tree, sample)
    if pred == label:
        correct_train += 1

print(f"Train accuracy: {correct_train/len(train_labels):.2%}")
print(f"Test accuracy:  {correct/len(test_labels):.2%}")