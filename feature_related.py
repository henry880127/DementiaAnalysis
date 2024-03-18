import numpy as np
from scipy import stats

class feature_extraction:
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def my_method(self):
        # Method code here
        pass

class feature_selection:
    def __init__(self, sampling_rate=500):
        self.sampling_rate = sampling_rate
    
    def fisher_score(self, x, y):
        # compute fisher score for each feature
        # x: data, {numpy array}, shape (n_samples, n_features)
        # y: labels, {numpy array}, shape (n_samples, )
        # return: {numpy array}, shape (n_features, )
        idx = np.argsort(y)
        y = y[idx]
        x = x[idx,:]
        n_samples, n_features = x.shape
        classes = np.unique(y)  # number of classes
        # compute fisher score for each feature
        score = np.zeros(n_features)
        for i in range(n_features):
            f = x[:,i]
            score[i] = self.fisher_score_per_feature(f, y, classes)
        return score
    
    def fisher_score_per_feature(self, f, y, classes):
        # compute fisher score of a single feature
        # f: feature, {numpy array}, shape (n_samples, )
        # y: labels, {numpy array}, shape (n_samples, )
        # classes: {int}, number of class
        # return: {float}, fisher score
        sb = 0
        # compute fisher score of the feature f
        for i, c in enumerate(classes):
            # find the index of feature belong to class c
            idx = np.where(y == c)[0]
            # compute the mean value of feature belong to class c
            f_c = f[idx]
            avg_c = np.mean(f_c)
            # compute the mean value of feature
            avg = np.mean(f)
            # compute the fisher score of feature f
            sb += len(idx) * np.square(avg_c-avg)
        sb = sb / len(y)

        sw = 0
        for i, c in enumerate(classes):
            # find the index of feature belong to class c
            idx = np.where(y == c)[0]
            # compute the mean value of feature belong to class c
            f_c = f[idx]
            avg_c = np.mean(f_c)
            si_c = np.sum(np.square(f_c - avg_c))
            si_c = si_c / (len(idx))
            si_c = si_c * (len(idx)/len(y)) # pi = len(idx)/len(y)
            sw += si_c
        score = sb / sw
        return score
    
    def pearson_coef(self,x,y):
        # compute pearson coefficient for each feature
        # x: data, {numpy array}, shape (n_samples, n_features)
        # y: numeric labels, {numpy array}, shape (n_samples, )
        # return: {numpy array}, shape (n_features, )
        n_samples, n_features = x.shape
        score = np.zeros(n_features)
        for i in range(n_features):
            f = x[:,i]
            score[i] = stats.pearsonr(f, y)[0]
            score[i] = np.abs(score[i])
        return score