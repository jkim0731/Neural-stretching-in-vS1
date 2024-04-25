import numpy as np
from sklearn.linear_model import LogisticRegression
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


def logireg_cross_validate(X, y, splits_inds, penalty='l2'):
    if len(X.shape)==1:
        X = X[:,np.newaxis]
    assert penalty in ['l1', 'l2']
    if penalty == 'l1':
        solver = 'liblinear'
    else:
        solver = 'lbfgs'
    num_splits = len(splits_inds)
    accuracies = []
    coeffs = []
    for si in range(num_splits):
        test_inds = splits_inds[si]
        train_inds = np.setdiff1d(np.arange(len(y)), test_inds)
        X_train = X[train_inds,:]
        y_train = y[train_inds]
        X_test = X[test_inds.astype(int),:]
        y_test = y[test_inds.astype(int)]
        clf = LogisticRegression(penalty=penalty, solver=solver)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
        coef = np.hstack([clf.intercept_.reshape(1,1), clf.coef_])
        coeffs.append(coef)
    return accuracies, coeffs


def get_logireg_results(X, y, num_split=4, num_repeat=100, penalty='l2',
                        min_num_trials_in_split=15):
    if X.shape[0] < 60:
        num_split = np.int(X.shape[0] / min_num_trials_in_split)
    all_mean_accuracy = []
    all_coeffs=[]
    for ri in range(num_repeat):
        splits_inds = stratify_random_split(np.arange(len(y)), y, num_splits=num_split)
        accuracies, coeffs = logireg_cross_validate(X, y, splits_inds, penalty=penalty)
        accuracy = np.mean(accuracies)
        coeff = np.mean(coeffs, axis=0)
        all_mean_accuracy.append(accuracy)
        all_coeffs.append(coeff)
    return np.mean(all_mean_accuracy), np.mean(coeffs, axis=0)


def get_shuffle_logireg_results(X, y, num_split=4, num_shuffle=100, penalty='l2'):
    splits_inds = stratify_random_split(np.arange(len(y)), y, num_splits=num_split)    
    shuffle_accuracies = []
    shuffle_coeffs = []
    for si in range(num_shuffle):
        shuffle_y = np.random.permutation(y)
        accuracies, coeffs = logireg_cross_validate(X, shuffle_y, splits_inds, penalty=penalty)
        accuracy = np.mean(accuracies)
        coeff = np.mean(coeffs, axis=0)
        shuffle_accuracies.append(accuracy)
        shuffle_coeffs.append(coeff)
    mean_shuffle_accuracy = np.mean(shuffle_accuracies)
    mean_shuffle_coeff = np.mean(shuffle_coeffs, axis=0)
    return mean_shuffle_accuracy, mean_shuffle_coeff