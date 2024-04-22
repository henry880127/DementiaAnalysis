# classifying the data using LDA and LOO-CV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from tqdm import tqdm, trange
import os
from feature_related import feature_selection   # Import the feature_selection module
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np

class Model_ML:
    def __init__(self, data, info, **kwargs):      
        self.df_accCV = list()
        self.data = data
        self.info = info

        # use **kwargs to set the new value of below args.
        self.folder_path_LDA = '../Results/Classification/LDA/10Fold_test/'
        self.folder_path_SVM = '../Results/Classification/SVM/10Fold_test/'
        self.param_grid_LDA = {'solver': ['svd']}
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])


    def CV_LDA(self):
        # Data preprocessing
        self.data_preprocessing(self.data, self.info)
        self.data_split_and_fisher()

        self.lda = LinearDiscriminantAnalysis()
        self.param_grid = self.param_grid_LDA

        folder_path = self.folder_path_LDA
        if not os.path.exists(folder_path):     os.makedirs(folder_path)

        max_dim = int(self.label_train_valid.shape[0]/2)  # Define the maximum number of features
        for idx_AOFI in tqdm(range(1, max_dim + 1)):
            data_CV = self.data_train_valid[:, self.fisher_idx[:idx_AOFI]]

            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=self.lda, param_grid=self.param_grid, cv=10, n_jobs=7)

            # Train the model with gridsearchCV
            grid_search.fit(data_CV, self.label_train_valid)

            # Get the best parameters and best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            # Get the best model
            best_model = grid_search.best_estimator_
            test_score = best_model.score(self.data_test[:, self.fisher_idx[:idx_AOFI]], self.label_test)

            # Append the loop index and accCV to the dataframe
            self.df_accCV.append({'#features': idx_AOFI, 'best_CVscore': best_score, 'best_params': best_params, 'test_score': test_score})

            # Save the best model to a file
            model_folder = os.path.join(folder_path, 'Models/')
            if not os.path.exists(model_folder):
                os.makedirs(model_folder)
            model_file = os.path.join(model_folder, 'best_model_' + str(idx_AOFI) + '.joblib')
            dump(best_model, model_file)
            pd.Series(self.fisher_idx[:idx_AOFI]).to_csv(model_file + '_features_idx.csv', index=False)
        # Save the dataframe to csv
        df_accCV = pd.DataFrame(self.df_accCV)
        df_accCV.to_csv(os.path.join(folder_path, 'df_accCV.csv'), index=False)

        # Create a dictionary with 'fisher_idx' and 'feature_type' as keys
        fisher_idx_DF = {'fisher_idx': self.fisher_idx, 'feature_type': self.data['train'].columns[self.fisher_idx]}
        fisher_idx_DF = pd.DataFrame(fisher_idx_DF)
        fisher_idx_DF.to_csv(os.path.join(folder_path, 'fisher_idx_series.csv'), index=False)

    def CV_SVM(self):
        # Data preprocessing
        self.data_preprocessing(self.data, self.info)
        self.data_split_and_fisher()

        # Create the SVM classifier
        svm = SVC()

        # Define the parameter grid for GridSearchCV
        gamma_range = np.linspace(-100,100,41) #-100,-95,...,95,100
        gamma_range = 1.05**gamma_range # 1.05^-100,1.05^-95,...,1.05^95,1.05^100
        gamma_range = 1 / 2*(np.square(gamma_range))  # gamma = 1 / (2*sigma)^2, based on the SVC documentation
        gamma_range = gamma_range.tolist()
        C_range = [1, 10, 100, 500, 1000]
        param_grid = {'C': C_range, 'gamma': gamma_range, 'kernel': ['linear','rbf']}


        # Create the folder if it does not exist
        folder_path = self.folder_path_SVM
        if not os.path.exists(folder_path):    os.makedirs(folder_path)

        df_accCV =list()
        max_dim = int(self.label_train_valid.shape[0]/2)  # Define the maximum number of features
        for idx_AOFI in tqdm(range(1, max_dim + 1)):
            data_CV = self.data_train_valid[:, self.fisher_idx[:idx_AOFI]]
            
            # Create the GridSearchCV object
            grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=10, n_jobs=7)

            # Train the model with gridsearchCV
            grid_search.fit(data_CV, self.label_train_valid)

            # Get the best parameters and best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            # Get the best model
            best_model = grid_search.best_estimator_
            test_score = best_model.score(self.data_test[:, self.fisher_idx[:idx_AOFI]], self.label_test)
            
            # Append the loop index and accCV to the dataframe
            df_accCV.append({'#features': idx_AOFI, 'best_CVscore': best_score, 'best_params': best_params, 'test_score': test_score})
            
            # Save the best model to a file
            model_folder = os.path.join(folder_path, 'Models/') # Specify the folder path to save models
            if not os.path.exists(model_folder):    os.makedirs(model_folder) # Create the folder if it does not exist
            model_file = os.path.join(model_folder, 'best_model_'+str(idx_AOFI)+'.joblib')
            dump(best_model, model_file)  # Save the best model to a file
            pd.Series(self.fisher_idx[:idx_AOFI]).to_csv(model_file+'_features_idx.csv', index=False)  # Save the best features index to a file

        # Save the dataframe to csv
        df_accCV = pd.DataFrame(df_accCV) # Convert the list to a dataframe
        df_accCV.to_csv(os.path.join(folder_path, 'df_accCV.csv'), index=False)

        # Create a dictionary with 'fisher_idx' and 'feature_type' as keys
        fisher_idx_DF = {'fisher_idx': self.fisher_idx, 'feature_type': self.data['train'].columns[self.fisher_idx]}
        fisher_idx_DF = pd.DataFrame(fisher_idx_DF) # Convert the dictionary to a DataFrame
        fisher_idx_DF.to_csv(os.path.join(folder_path, 'fisher_idx_series.csv'), index=False) # Save the DataFrame to a .csv file

    def data_split_and_fisher(self):
        # Train & Valid data organizing
        self.data_train_valid = np.concatenate([self.data_ndarrays['train'], self.data_ndarrays['valid']], axis=0)
        self.label_train_valid = np.concatenate([self.label_ndarrays_CInonCI['train'], self.label_ndarrays_CInonCI['valid']], axis=0)

        # Independent test data organizing
        self.data_test = self.data_ndarrays['test']
        self.label_test = self.label_ndarrays_CInonCI['test']

        # Feature selection - filter_based
        f_selection = feature_selection()
        fisher_scores = f_selection.fisher_score(self.data_train_valid, self.label_train_valid)
        self.fisher_idx = np.argsort(fisher_scores)[::-1] # sort in descending order

        return self.data_train_valid, self.label_train_valid, self.data_test, self.label_test, self.fisher_idx


    def data_preprocessing(self, data, info):
        # Dataframe to numpy array in each dictionary
        data_ndarrays = {}
        for key, value in data.items():
            # Detect if the key is 'Info' and skip it
            if key != 'Info':
                # Drop the 'ID' and 'Task' columns if they exist
                if 'ID' in data[key].columns:
                    data[key].drop('ID', axis=1, inplace=True)
                if 'Task' in data[key].columns:
                    data[key].drop('Task', axis=1, inplace=True)  
                data_ndarrays[key] = value.values
            else:
                data_ndarrays[key] = value

        label_ndarrays = {}
        label_ndarrays_CInonCI = {}
        for key, value in info.items():
            if key != 'Info':
                label_ndarrays[key] = value['Label'].values
                label_ndarrays_CInonCI[key] = value['Label'].values
            else:
                label_ndarrays[key] = value
                label_ndarrays_CInonCI[key] = value

        # Replace 2 with 1 in label_ndarrays
        label_ndarrays_CInonCI = {key: np.where(value == 2, 1, value) for key, value in label_ndarrays_CInonCI.items()}


        self.data_ndarrays = data_ndarrays
        self.label_ndarrays = label_ndarrays
        self.label_ndarrays_CInonCI = label_ndarrays_CInonCI
        return data_ndarrays, label_ndarrays, label_ndarrays_CInonCI
