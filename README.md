# DementiaAnalysis

 Analysis related codes! The dataset is unavailable to access by others.


Created and edited by WeiHung Kuan, 2024/07/12.


## Interactive_interface.ipynb

Please refer to "Interactive_interface.ipynb" for machine learning related analysis.

You can treat "Interactive_interface.ipynb" as the main program.

Note that SFS is implemented through "mlxtend" library.


## analysis_EEGNet.ipynb

EEGNet_related analysis. Include multiple preprocess workflow.

DL models were implemented using tensorflow library, it's free to transfer it into pytorch

In the tesis, I've analyzed EEGNet, EEGTCNet, DeepConvNet, SAEEGNet. The analysis_{modelname}.ipynb of the above models is similar to "analysis_EEGNet.ipynb".


## Classes

"Model_ML.py" includes several pipeline to conduct nalysis. (abondoned however)

"feature_related.py" includes the classed or method related to feature selection (Fisher's criterion, RFE, etc.)

"Data_preprocess_{modelname}" are the classes utilized to conduct normalization, preprocess on EEG signal, corresponds to specific model.

"utils.py" is the code that conduct "gridSearchCV within SFS", which conduct gridsearch in every feature subset in SFS. (However, you can find the same code in "Interactive_interface.ipynb")

"PSDCNN_function.py" is the code related to [1]

"conn_utils" is the code for the experiment inspired by [2] (However, it is not mentioned in the thesis or any presentation)

"Model_2outputs.py" is the abandoned experiment.

"{modelname}_function.py" or "{modelname}_utils.py" are the classes help users to implement these DL models.



[1] C. Ieracitano, N. Mammone, A. Bramanti, A. Hussain, and F. C. Morabito,“A Convolutional Neural Network approach for classification of dementia stages based on 2D-spectral representation of EEG recordings,” Neurocomputing, vol. 323, pp. 96–107, Jan. 2019, doi: 10.1016/j.neucom.2018.09.071.

[2]	M. A. H. Akhand, M. A. Maria, M. A. S. Kamal, and K. Murase, "Improved EEG-based emotion recognition through information enhancement in connectivity feature map," Sci Rep, vol. 13, no. 1, p. 13804, Aug 23 2023
