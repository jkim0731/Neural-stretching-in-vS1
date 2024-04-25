import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import accuracy_score


def random_split(inds, num_split=4):
    inds = np.random.choice(inds, len(inds), replace=False)
    split = []
    group_ids = np.arange(len(inds)) % num_split
    for gi in range(num_split):
        split.append(inds[group_ids==gi])
    return split


def stratify_random_split(inds, stratify_class, num_splits=4):
    assert len(inds) == len(stratify_class)
    classes = np.unique(stratify_class)
    num_classes = len(classes)
    ci = 0
    splits = [[] for i in range(num_splits)]
    for ci in range(num_classes):
        class_inds = np.where(stratify_class==classes[ci])[0]
        split_temp = random_split(inds[class_inds], num_split=num_splits)
        for gi in range(num_splits):
            splits[gi] = np.concatenate([splits[gi], split_temp[gi]]).astype(int)
    return splits


def lda_cross_validate(X, y, splits_inds):
    if len(X.shape)==1:
        X = X[:,np.newaxis]
    num_splits = len(splits_inds)
    accuracies = []
    for si in range(num_splits):
        test_inds = splits_inds[si]
        train_inds = np.setdiff1d(np.arange(len(y)), test_inds)
        X_train = X[train_inds,:]
        y_train = y[train_inds]
        X_test = X[test_inds.astype(int),:]
        y_test = y[test_inds.astype(int)]
        lda = LDA()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def get_lda_accuracies(X, y, num_split=4, num_repeat=100):
    if X.shape[0] < 60:
        num_split = np.int(X.shape[0] / 15)
    all_mean_accuracy = []
    for ri in range(num_repeat):
        splits_inds = stratify_random_split(np.arange(len(y)), y, num_splits=num_split)
        accuracy = np.mean(lda_cross_validate(X, y, splits_inds))        
        all_mean_accuracy.append(accuracy)
    return np.mean(all_mean_accuracy)


def get_shuffle_lda_accuracies(X, y, num_split=4, num_shuffle=100):
    splits_inds = stratify_random_split(np.arange(len(y)), y, num_splits=num_split)    
    shuffle_accuracies = []
    for si in range(num_shuffle):
        shuffle_y = np.random.permutation(y)
        shuffle_accuracies.append(lda_cross_validate(X, shuffle_y, splits_inds))
    shuffle_accuracies = np.array(shuffle_accuracies)
    mean_shuffle_accuracy = np.mean([np.mean(sa) for sa in shuffle_accuracies])
    
    return mean_shuffle_accuracy