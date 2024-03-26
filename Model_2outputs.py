from keras.layers import Input, Dense, Dropout
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
import time
import os
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score, mean_absolute_error, r2_score


# Ref: https://machinelearningmastery.com/deep-learning-models-for-multi-output-regression/
# Ref: https://machinelearningmastery.com/neural-network-models-for-combined-classification-and-regression/
# Don't know why, but when setting the loss function, follow the following order: regression, classification
class Model_2outputs():
    def __init__(self,
                 input_dim,
                 n_classes,
                 optimizer=Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                 lr=0.000001,
                 model_name='Model_2outputs',
                 epochs=100,
                 batch_size=32,
                 log_path='Models/logs',
                 loss = ['mse', 'sparse_categorical_crossentropy'],
                 **kwargs):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.optimizer = optimizer
        # self.optimizer.lr = lr
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_path = log_path
        self.loss = loss

        # use **kwargs to set the new value of below args.
        self.f1_average = 'binary' if self.n_classes == 2 else 'macro'
        self.nNeurons_common = 64
        self.nNeurons_net1 = [64,32]
        self.nNeurons_net2 = [64,32]
        self.dropout_rate = 0.5
        self.norm_rate = 0.25
        self.dropout_rate = 0.5
        self.monitor = 'val_loss' # save_best_only based on this monitor

        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    def build(self):
        input_layer = Input(shape=(self.input_dim,))

        # common_layer = Dense(self.nNeurons_common, activation='relu')(input_layer)
        common_layer = Dense(20, activation='relu', kernel_initializer='he_normal')(input_layer)
        common_layer = Dense(10, activation='relu', kernel_initializer='he_normal')(common_layer)
    
        # sub-network 1  for classification
        # sub_net1 = Dense(self.nNeurons_net1[0], activation='relu')(common_layer)
        # sub_net1 = Dense(self.nNeurons_net1[1], activation='relu')(sub_net1)
        # sub_net1 = Dropout(self.dropout_rate)(sub_net1)
        # sub_net1_output = Dense(self.n_classes, activation='softmax')(sub_net1)
        sub_net1_output = Dense(self.n_classes, activation='softmax')(common_layer) #test open source's structure from ref

        # sub-network 2  for regrssion
        # sub_net2 = Dense(self.nNeurons_net2[0], activation='relu')(common_layer)
        # sub_net2 = Dense(self.nNeurons_net2[0], activation='relu')(sub_net2)
        # sub_net2 = Dropout(self.dropout_rate)(sub_net2)
        # sub_net2_output = Dense(1, activation='relu')(sub_net2)
        sub_net2_output = Dense(1, activation='linear')(common_layer) #test open source's structure from ref

        # Combine both sub-network outputs
        combined_output = [sub_net2_output, sub_net1_output]

        # Define the model with input and combined output
        model = Model(inputs=input_layer, outputs=combined_output)
        
        # Compile the model
        # Don't know why, but when setting the loss function, follow the following order: regression, classification
        model.compile(optimizer=self.optimizer, loss=self.loss)

        # plot the model
        file_path = self.log_path + '/' + self.model_name + '/' + self.model_name + '.h5'
        plot_model(model, to_file=self.log_path + '/model.png', show_shapes=True)

        # Print model summary
        model.summary()
        
        self.model = model
        return model

    def fit(self, X_train,  y_train_reg, y_train_cat, X_val, y_val_cat, y_val_reg):
        # Define the callbacks
        file_path = self.log_path + '/' + self.model_name + '/' + self.model_name + '.h5'
        if not os.path.exists(self.log_path + '/' + self.model_name):
            os.makedirs(self.log_path + '/' + self.model_name)
        checkpointer = ModelCheckpoint(monitor=self.monitor, filepath=file_path, 
                                       verbose=1, save_best_only=True, save_weights_only=True)
        
        model = self.build()
        history = model.fit(X_train,[y_train_reg, y_train_cat], 
                            epochs=self.epochs, batch_size=self.batch_size,
                            validation_data=(X_val, [y_val_cat, y_val_reg]), verbose=1,
                            callbacks=[checkpointer])
        
        return history
    
    def score(self, X_test, y_test_reg, y_test_cat):
        model = self.build() # Build the model
        model.load_weights(self.log_path + '/' + self.model_name + '/' + self.model_name + '.h5') # Load the best model
        model.compile(optimizer=self.optimizer, loss=self.loss) # Compile the model

        start = time.time()  # Record the start time
        [y_pred_reg, y_pred_cat] = model.predict(X_test) # Predict the test data
        end = time.time() # Record the end time

        # performance evaluation of classification(category)
        y_pred_cat = np.argmax(y_pred_cat, axis=-1).astype(int) # Get the predicted category
        acc = accuracy_score(y_test_cat, y_pred_cat)
        print('Accuracy: %.3f' % acc)

        # performance evaluation of regression
        print(y_test_reg.shape, y_pred_reg.shape)
        score = r2_score(y_test_reg, y_pred_reg) # Calculate the mean absolute error
        print('r2_score: %.3f' % score)
       
        loss = model.evaluate(x=X_test, y=[y_test_reg, y_test_cat], batch_size=self.batch_size)

        evaluation = {'loss': loss, 
                      'accuracy': acc,
                      'r2_score': score, 
                      'prediction_time': end-start}
        
        Y = {'y_true': [y_test_cat, y_test_reg], 
             'y_pred': [y_pred_cat, y_pred_reg]}
        return Y, evaluation