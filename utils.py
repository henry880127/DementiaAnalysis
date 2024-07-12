import numpy as np
import pandas as pd

def scaling(datx):
    ndatx = []
    for dx in datx:
        mean, std = np.average(dx, axis=0), np.std(dx, axis=0)
        ndatx.append((dx-mean)/std)
    return np.array(ndatx)

class sequential_feture_selection_gridSearch():
    def __init__(self, gridSearchCV, k_features=None, forward=True):
        self.gridSearchCV = gridSearchCV
        self.k_features = k_features
        self.forward = forward
        
        # create a list to store the chosen features (to eliminate or add features to the model)
        self.chosen_feature_idx = list()
        self.remain_feature_idx = list()
        
        # create a dict to store the metrics of all iterations
        self.metrics = dict()
        self.metrics['CV_score'] = list()
        self.metrics['feature_idx'] = list()
        self.metrics['best_params'] = list()


    def iteratly_fit_remain_feature(self, X, y):
        # Forward selection
        scores = list()
        iteratly_best_params = list()
        if self.forward:
            print('len(self.remain_feature_idx):',len(self.remain_feature_idx))
            for idx, feature_idx in enumerate(self.remain_feature_idx):
                self.gridSearchCV.fit(X[:, self.chosen_feature_idx+[feature_idx]], y) # Fit the gridSearchCV with the chosen features
                scores.append(self.gridSearchCV.best_score_) # Append the CV score of the best_estimator to the list
                iteratly_best_params.append(self.gridSearchCV.best_params_)

            scores = np.array(scores) # Convert the list to a numpy array
            best_idx = np.argmax(scores) # Get the index of the best score
            self.chosen_feature_idx.append(self.remain_feature_idx[best_idx]) # Append the best feature to the chosen_feature_idx
            self.remain_feature_idx.pop(best_idx) # Remove the best feature from the remain_feature_idx

            # Store the metrics of this iteration
            self.metrics['CV_score'].append(scores[best_idx])
            self.metrics['feature_idx'].append(self.chosen_feature_idx.copy())
            self.metrics['best_params'].append(iteratly_best_params[best_idx])

        # Backward selection
        else:
            print('self.remain_feature_idx:',self.remain_feature_idx)
            for idx, feature_idx in enumerate(self.remain_feature_idx):
                if len(self.remain_feature_idx) == 1:
                    self.gridSearchCV.fit(X[:, self.remain_feature_idx], y)
                    scores.append(self.gridSearchCV.best_score_)
                    iteratly_best_params.append(self.gridSearchCV.best_params_)
                    break
                
                iteratly_remain_feature_idx = self.remain_feature_idx.copy()
                iteratly_remain_feature_idx.remove(feature_idx)
                self.gridSearchCV.fit(X[:, iteratly_remain_feature_idx], y)
                scores.append(self.gridSearchCV.best_score_)
                iteratly_best_params.append(self.gridSearchCV.best_params_)

            scores = np.array(scores) # Convert the list to a numpy array
            worst_idx = np.argmin(scores) # Get the index of the worst score
            self.chosen_feature_idx.append(self.remain_feature_idx[worst_idx]) # Remove the worst feature from the chosen_feature_idx
            self.remain_feature_idx.pop(worst_idx) # Remove the worst feature from the remain_feature_idx

            # Store the metrics of this iteration
            self.metrics['CV_score'].append(scores[worst_idx])
            self.metrics['feature_idx'].append(self.remain_feature_idx.copy())
            self.metrics['best_params'].append(iteratly_best_params[worst_idx])
        

    def fit(self, X, y):
        # Initialize the remain_feature_idx
        self.remain_feature_idx = list(range(X.shape[1]))
        self.k_features = len(self.remain_feature_idx) if self.k_features is None else self.k_features

        if self.forward == False:
            self.gridSearchCV.fit(X, y)
            # Store the metrics of this iteration
            self.metrics['CV_score'].append(self.gridSearchCV.best_score_)
            self.metrics['feature_idx'].append(self.remain_feature_idx.copy())
            self.metrics['best_params'].append(self.gridSearchCV.best_params_)

        # Iterate the feature selection
        for idx in range(self.k_features):
            print('idx:', idx)
            self.iteratly_fit_remain_feature(X, y)
            if self.forward == False and len(self.remain_feature_idx) == 1:
                break

        metrics = pd.DataFrame(self.metrics)
        best_idx = np.argmax(metrics['CV_score'])
        print('best_idx:', best_idx)
        self.best_metric = metrics.iloc[best_idx]

        print('best_metric:', self.best_metric)

