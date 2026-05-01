import math
from collections import Counter

import math
from collections import Counter


def entropy(labels):
    n = len(labels)
    counts = Counter(labels)

    h = 0
    for count in counts.values():
        p = count / n
        h += p * math.log2(p)

    return -h


def information_gain(feature, labels):
    h_before = entropy(labels)
    n = len(labels)

    groups = {}
    for f, l in zip(feature, labels):
        groups.setdefault(f, []).append(l)

    h_after = 0
    for g in groups.values():
        h_after += (len(g) / n) * entropy(g)

    return h_before - h_after


def best_feature(features_dict, labels):
    ig_scores = {}
    for name, vals in features_dict.items():
        ig_scores[name] = information_gain(vals, labels)

    return max(ig_scores, key=ig_scores.get)


def build_tree(features_dict, labels, min_samples=2):
    # Điều kiện dừng 1: entropy = 0
    if entropy(labels) == 0:
        return {
            'leaf': labels[0]
        }

    # Điều kiện dừng 2: hết feature
    if len(features_dict) == 0:
        most_common = Counter(labels).most_common(1)[0][0]
        return {
            'leaf': most_common
        }

    # Điều kiện dừng 3: quá ít data
    if len(labels) < min_samples:
        most_common = Counter(labels).most_common(1)[0][0]
        return {
            'leaf': most_common
        }

    # Tìm best feature
    best = best_feature(features_dict, labels)

    # Split data theo từng nhánh
    branches = {}
    for i in range(len(labels)):
        value = features_dict[best][i]
        if value not in branches:
            branches[value] = {'features': {}, 'labels': []}
        branches[value]['labels'].append(labels[i])
        for name, vals in features_dict.items():
            if name != best:
                branches[value]['features'].setdefault(name, []).append(vals[i])

    # Đệ quy build_tree cho mỗi nhánh
    tree = {'feature': best, 'branches': {}}
    for value, data in branches.items():
        tree['branches'][value] = build_tree(data['features'], data['labels'], min_samples)

    return tree


def predict(tree, sample):
    # Nếu là leaf → trả về kết quả
    if 'leaf' in tree:
        return tree['leaf']

    # Lấy feature cần hỏi từ tree
    feature = tree['feature']
    branches = tree['branches']

    # Lấy giá trị của feature đó từ sample
    value = sample[feature]

    # Đi theo nhánh tương ứng → đệ quy
    new_tree = branches[value]
    return predict(new_tree, sample)

features_dict = {
    'Huyết áp': ['Cao', 'Cao', 'Cao', 'Thấp', 'Thấp', 'Thấp'],
    'Tuổi':     ['Trẻ', 'Già', 'Già', 'Trẻ',  'Già',  'Già' ]
}
labels = ['Có', 'Có', 'Có', 'Không', 'Không', 'Không']

# print(build_tree(features_dict, labels, min_samples=2))

tree = {
    'feature': 'Huyết áp',
    'branches': {
        'Cao':  {'leaf': 'Có'},
        'Thấp': {'leaf': 'Không'}
    }
}

sample = {'Huyết áp': 'Cao', 'Tuổi': 'Trẻ'}
# print(predict(tree, sample))  # kết quả?

def to_features_dict(samples):
    # samples = [{'sepal length': 'thap', 'sepal width': 'cao', ...}, ...]
    # output  = {'sepal length': ['thap', ...], 'sepal width': ['cao', ...], ...}
    result = {}
    for sample in samples:
        for name, val in sample.items():
            result.setdefault(name, []).append(val)
    return result