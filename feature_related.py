import numpy as np
from scipy import stats
import pandas as pd

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
    
    def remove_collinear_features(self, x, threshold):
        '''
        ref: https://stackoverflow.com/questions/29294983/how-to-calculate-correlation-between-all-columns-and-remove-highly-correlated-on
        Objective:
            Remove collinear features in a dataframe with a correlation coefficient
            greater than the threshold. Removing collinear features can help a model 
            to generalize and improves the interpretability of the model.

        Inputs: 
            x: features dataframe
            threshold: features with correlations greater than this value are removed

        Output: 
            dataframe that contains only the non-highly-collinear features
        '''

        x = pd.DataFrame(x)
        # Calculate the correlation matrix
        corr_matrix = x.corr()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the threshold
                if val >= threshold:
                    # Print the correlated features and the correlation value
                    print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                    drop_cols.append(col.values[0])

        # Drop one of each pair of correlated columns
        drops = set(drop_cols)
        x = x.drop(columns=drops)
        
        x = x.values
        return x, drops
    
from sklearn.svm import SVC
class RFECV_pseudo_sample():
    def __init__(self, n_pseudo_samples, fitted_estimator, n_features_to_select=0,  c=1.4826):
        self.n_features_to_select = n_features_to_select
        self.fitted_estimator = fitted_estimator
        self.params = fitted_estimator.get_params()
        self.n_pseudo_samples = n_pseudo_samples
        self.discarded_features_idx = list()
        self.remain_feature_idx = list()
        self.c = c

    def create_pseudo_samples(self, X):
        n_features = X.shape[1]
        
        # Create a list to store the pseudo samples
        pseudo_samples = []
        
        # Calculate the quantiles of all features
        quantiles = np.array([i/self.n_pseudo_samples for i in range(self.n_pseudo_samples)]) # create quantiles
        quantilized_features = np.quantile(X, quantiles, axis=0) # calculate quantiles of each feature
        
        # Calculate the median of all features
        median_features = np.median(X, axis=0)

        # Loop through each feature
        for i in range(n_features):

            # Create a copy of the data
            X_pseudo = np.array([median_features.copy() for i in range(self.n_pseudo_samples)])
            
            # convert specific feature to quantilized value
            X_pseudo[:,i] = quantilized_features[:,i]
            
            # Append the pseudo sample to the list
            pseudo_samples.append(X_pseudo)
        
        # Convert the list to a numpy array
        pseudo_samples = np.array(pseudo_samples)
        
        self.pseudo_samples_ = pseudo_samples
        return self.pseudo_samples_
    
    def iteratly_fit_remain_feature(self, pseudo_samples, X, y):
        iteratly_remain_feature_idx = self.remain_feature_idx.copy()    

        # fit the model on the remaining features
        self.fitted_estimator.fit(X[:, iteratly_remain_feature_idx], y)
        print('iteratly_remain_feature_idx:',iteratly_remain_feature_idx)

        # Calculate the feature importance of the pseudo samples
        MAD = np.zeros(len(self.remain_feature_idx))
        for idx, feature_idx in enumerate(self.remain_feature_idx):
            # Access the decision values of the pseudo samples
            print(pseudo_samples.shape)
            tmp = pseudo_samples[feature_idx, :, :]
            D = self.fitted_estimator.decision_function(tmp[:, iteratly_remain_feature_idx])
            print(tmp[:, iteratly_remain_feature_idx].shape)
            
            # Calculate the median absolute deviation (MAD)
            MAD[idx] = np.median(np.absolute(D - np.median(D)))*self.c
        
        # Sort the feature importance
        print('MAD:',MAD)
        discard_idx = np.argmin(MAD)
        print('discard_idx:',discard_idx)
        self.discarded_features_idx.append(self.remain_feature_idx[discard_idx]) 
        self.remain_feature_idx.pop(discard_idx) # Remove the worst feature from the remain_feature_idx
    
        
    def fit(self, X, y):
        self.remain_feature_idx = list(range(X.shape[1]))
        
        # Create pseudo samples
        pseudo_samples = self.create_pseudo_samples(X)

        # Loop through each iteration
        for i in range(X.shape[1] - self.n_features_to_select):
            # Fit the model on the remaining features
            self.iteratly_fit_remain_feature(pseudo_samples, X, y)
        
        self.ranked_features_ascending_ = self.discarded_features_idx.copy()
        self.ranked_features_ascending_ += self.remain_feature_idx
        self.ranked_features_ascending_ = self.ranked_features_ascending_[::-1]

class RFE_pseudo_sample():
    def __init__(self, n_pseudo_samples, fitted_estimator, n_features_to_select=0,  c=1.4826):
        self.n_features_to_select = n_features_to_select
        self.fitted_estimator = fitted_estimator
        self.params = fitted_estimator.best_estimator_.get_params()
        self.n_pseudo_samples = n_pseudo_samples
        self.discarded_features_idx = list()
        self.remain_feature_idx = list()
        self.c = c

    def create_pseudo_samples(self, X):
        n_features = X.shape[1]
        
        # Create a list to store the pseudo samples
        pseudo_samples = []
        
        # Calculate the quantiles of all features
        quantiles = np.array([i/self.n_pseudo_samples for i in range(self.n_pseudo_samples)]) # create quantiles
        quantilized_features = np.quantile(X, quantiles, axis=0) # calculate quantiles of each feature
        
        # Calculate the median of all features
        median_features = np.median(X, axis=0)

        # Loop through each feature
        for i in range(n_features):

            # Create a copy of the data
            X_pseudo = np.array([median_features.copy() for i in range(self.n_pseudo_samples)])
            
            # convert specific feature to quantilized value
            X_pseudo[:,i] = quantilized_features[:,i]
            
            # Append the pseudo sample to the list
            pseudo_samples.append(X_pseudo)
        
        # Convert the list to a numpy array
        pseudo_samples = np.array(pseudo_samples)
        
        self.pseudo_samples_ = pseudo_samples
        return self.pseudo_samples_
    
    def iteratly_fit_remain_feature(self, pseudo_samples, X, y):
        iteratly_remain_feature_idx = self.remain_feature_idx.copy()    

        # fit the model on the remaining features
        estimator = self.fitted_estimator.best_estimator_

        estimator.fit(X[:, iteratly_remain_feature_idx], y)
        # print('iteratly_remain_feature_idx:',iteratly_remain_feature_idx)

        # Calculate the feature importance of the pseudo samples
        MAD = np.zeros(len(self.remain_feature_idx))
        for idx, feature_idx in enumerate(self.remain_feature_idx):
            # Access the decision values of the pseudo samples
            # print(pseudo_samples.shape)
            tmp = pseudo_samples[feature_idx, :, :]
            D = estimator.decision_function(tmp[:, iteratly_remain_feature_idx])
            # print(tmp[:, iteratly_remain_feature_idx].shape)
            
            # Calculate the median absolute deviation (MAD)
            MAD[idx] = np.median(np.absolute(D - np.median(D)))*self.c
        
        # Sort the feature importance
        print('MAD:',MAD)
        discard_idx = np.argmin(MAD)
        print('discard_idx:',discard_idx)
        self.discarded_features_idx.append(self.remain_feature_idx[discard_idx]) 
        self.remain_feature_idx.pop(discard_idx) # Remove the worst feature from the remain_feature_idx
    
        
    def fit(self, X, y):
        self.remain_feature_idx = list(range(X.shape[1]))
        
        # Create pseudo samples
        pseudo_samples = self.create_pseudo_samples(X)

        # Loop through each iteration
        for i in range(X.shape[1] - self.n_features_to_select):
            # Fit the model on the remaining features
            self.iteratly_fit_remain_feature(pseudo_samples, X, y)
        
        self.ranked_features_ascending_ = self.discarded_features_idx.copy()
        self.ranked_features_ascending_ += self.remain_feature_idx
        self.ranked_features_ascending_ = self.ranked_features_ascending_[::-1]