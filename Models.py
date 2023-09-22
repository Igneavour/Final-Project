import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
from timeit import timeit

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten, LSTM
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import svm

tf.random.set_seed(42)
np.random.seed(42)

class SVM:
    def __init__(self, X, y): 
        
        self.X = X 
        self.y = y
            
    def preprocess_data(self):
        
        # Split data into train and test before scaling 
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify = self.y)
        
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
        
    def run_model(self, iterations):
        
        results = pd.DataFrame(columns=["acc","rec","prec","far","f1"])    
        
        for i in range(iterations):
            
            X_train, X_test, y_train, y_test = self.preprocess_data()

            model = svm.SVC()
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            acc = metrics.accuracy_score(y_test,y_pred)
            rec = metrics.recall_score(y_test,y_pred,average='weighted')
            prec = metrics.precision_score(y_test,y_pred,average='weighted')
            f1 = metrics.f1_score(y_test,y_pred,average='weighted')
            
            # Determine false alarm rate using the confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)
            FAR = FP/(FP+TN)
            
            model_results = {'acc':acc, 'rec':rec, 'prec':prec, 'far':FAR.mean(), 'f1':f1}
            results = results.append(model_results, ignore_index= True)
            print(f"Iteration {i} \n Model accuracy:", acc, "\n")          
        self.acc = results.mean()[0], results.std()[0]
        self.rec = results.mean()[1], results.std()[1]
        self.prec = results.mean()[2], results.std()[2]
        self.far = results.mean()[3], results.std()[3]  
        self.f1 = results.mean()[4], results.std()[4]
        
class MLP:
    def __init__(self, X, y):  
        
        self.X = X 
        self.y = y
            
    def preprocess_data(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify = self.y)
            
        # Scale data into suitable form
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
        
        return X_train, y_train, X_test, y_test
        
    def train_model(self):
        
        # Split data into training and testing set
        X_train, y_train, X_test, y_test = self.preprocess_data()

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        model = Sequential()
       
        early_stop = EarlyStopping(monitor='val_loss', mode = 'auto', 
                                   patience = 5 , verbose =0, restore_best_weights=True)

        model.add(Dense(512, kernel_regularizer=regularizers.L1L2(l1=0.00001, l2=0.0001), 
                                       bias_regularizer=regularizers.L2(0.0001), 
                                       activation='relu', input_shape=(X_train.shape[1:])))
        model.add(Dropout(0.4))
        model.add(Dense(128, kernel_regularizer=regularizers.L1L2(l1=0.00001, l2=0.0001), 
                                bias_regularizer=regularizers.l2(0.0001), activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(64, kernel_regularizer=regularizers.L1L2(l1=0.00001, l2=0.0001), 
                                bias_regularizer=regularizers.l2(0.0001), activation='relu'))
        model.add(Dense(7, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(x = X_train, y = y_train, epochs = 100, batch_size = 256, verbose = 0,
                  validation_data = (X_val,y_val),callbacks = [early_stop])
        
        return model, X_test, y_test
        
 
    def run_model(self, iterations):
        
        results = pd.DataFrame(columns=["acc","rec","prec","far","f1","Train Time","Test Time"])    
        self.preprocess_data()
        
        for i in range(iterations):
            
            # Timing the time taken for training and testing model
            print(f"Iteration {i} \n Model training in progress.")
            training_time = timeit(lambda: self.train_model(), number = 1)
            
            model, X_test, y_test = self.train_model()
            print(f" Model testing in progress.")
            testing_time = timeit(lambda: model.predict(X_test), number = 1)          
            y_pred = np.argmax(model.predict(X_test), axis=-1)
            
            acc = metrics.accuracy_score(y_test,y_pred)
            prec = metrics.precision_score(y_test,y_pred,average='weighted')
            rec = metrics.recall_score(y_test,y_pred,average='weighted')
            f1 = metrics.f1_score(y_test,y_pred,average='weighted')
            
            # Determine false alarm rate using the confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)
            FAR = FP/(FP+TN)
            
            model_results = {'acc':acc, 'rec':rec, 'prec':prec, 'far':FAR.mean(), 'f1':f1,
                             'Train Time':training_time,'Test Time':testing_time}
            results = results.append(model_results, ignore_index= True)
            print(f" Model {i} accuracy:", acc, "\n") 
        self.acc = results.mean()[0], results.std()[0]
        self.rec = results.mean()[1], results.std()[1]
        self.prec = results.mean()[2], results.std()[2]
        self.far = results.mean()[3], results.std()[3]  
        self.f1 = results.mean()[4], results.std()[4]
        self.train_time = results.mean()[5], results.std()[5]
        self.test_time = results.mean()[6], results.std()[6]
        
class CNN:
    def __init__(self, X, y, image_dim):
        
        self.image_dim = image_dim

        # Add two extra features for 2D image inputs 
        a = np.zeros((len(X),1))
        b = np.zeros((len(X),1))
        self.X = np.hstack((X,a,b))
        self.y = y
            
    def preprocess_data(self):
        
        # Split data into train and test split 
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify = self.y)

        # Scale data into suitable form
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)

        # Encoding labels into suitable form for model
        y_train = pd.get_dummies(y_train).values
        y_test = pd.get_dummies(y_test).values       

        # Reshape the data into 2D arrays for CNN input
        X_train = X_train.reshape(len(X_train), self.image_dim, self.image_dim,1)
        X_test = X_test.reshape(len(X_test), self.image_dim, self.image_dim,1) 

        return X_train, y_train, X_test, y_test
        
    def show_attack_matrix(self, types):
        X_train, y_train, X_test, y_test = self.preprocess_data()
        y_train_series = pd.Series(np.argmax(y_train,axis=1))
        fig, axs = plt.subplots(nrows =1 , ncols = (len(types)), figsize= (15,6))
        for t in types.keys():
            X_image = X_train[y_train_series[y_train_series==t].index[0]]
            axs[t].imshow(X_image, cmap='Greys_r')
            axs[t].set_title(types[t])
            axs[t].axis('off')
        plt.show()
        
    def train_model(self):
        
        # Split data into training and testing set, scale it and encode labels 
        X_train, y_train, X_test, y_test = self.preprocess_data()

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify= y_train)

        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=(2,2), input_shape=(self.image_dim,self.image_dim,1), padding='same', activation='relu'))
        model.add(BatchNormalization())       
        model.add(MaxPool2D(pool_size=(2, 2)))      
        model.add(Conv2D(filters=32, kernel_size=(2,2), padding='same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))    
        model.add(Flatten())        
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.2))        
        model.add(Dense(y_train.shape[1],activation = "softmax"))
        model.compile(loss="categorical_crossentropy", optimizer = 'adam')        
        early_stop = EarlyStopping(monitor='val_loss', mode = 'auto', 
                                   patience = 5 , verbose = 0, restore_best_weights=True)        
        model.fit(x = X_train, y = y_train, epochs = 100, batch_size = 256, verbose = 0, 
                  callbacks = [early_stop], validation_data = (X_val,y_val))
        
        return model, X_test, y_test
        
 
    def run_model(self, runs):
        results = pd.DataFrame(columns=["acc","rec","prec","far","f1","Train Time","Test Time"])
        reports = []
        
        # Train and evaluate model 
        for i in range(runs):
            
            # Timing the time taken for training and testing model
            print(f"Iteration {i} \n Model training in progress.")
            training_time = timeit(lambda: self.train_model(), number = 1)
            
            model, X_test, y_test = self.train_model() 
            print(f" Model testing in progress.")
            testing_time = timeit(lambda: model.predict(X_test), number = 1)          
            y_pred = np.argmax(model.predict(X_test), axis=-1)                                    
            y_org = np.argmax(y_test,axis=1)
            
            acc = metrics.accuracy_score(y_org,y_pred)
            prec = metrics.precision_score(y_org,y_pred,average='weighted')
            rec = metrics.recall_score(y_org,y_pred,average='weighted')
            f1 = metrics.f1_score(y_org,y_pred,average='weighted')
            
            confusion_matrix = metrics.confusion_matrix(y_org,y_pred)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)
            FAR = FP/(FP+TN)
            
            model_results = {'acc':acc, 'rec':rec, 'prec':prec, 'far':FAR.mean(), 'f1':f1,
                             'Train Time':training_time,'Test Time':testing_time}
            results = results.append(model_results, ignore_index= True)
            print(f" Model {i} accuracy:", acc, "\n") 
        self.acc = results.mean()[0], results.std()[0]
        self.rec = results.mean()[1], results.std()[1]
        self.prec = results.mean()[2], results.std()[2]
        self.far = results.mean()[3], results.std()[3]  
        self.f1 = results.mean()[4], results.std()[4]
        self.train_time = results.mean()[5], results.std()[5]
        self.test_time = results.mean()[6], results.std()[6]
        
class self_LSTM:
    def __init__(self, X, y):
        
        self.attack_types = 7

        # Add two extra features for 2D image inputs 
        a = np.zeros((len(X),1))
        b = np.zeros((len(X),1))
        self.X = np.hstack((X,a,b))
        self.y = y

    def preprocess_data(self):
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify = self.y)

        # Scale Data 
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
            
        # Reshape the data into 3D input arrays which include a time dimension as input for LSTM
        X_train = X_train.reshape(len(X_train), 1, X_train.shape[1])
        X_test = X_test.reshape(len(X_test), 1, X_test.shape[1]) 

        return X_train, y_train, X_test, y_test
      
        
    def train_model(self):
        
        # Split data into training and testing set, scale it and encode labels 
        X_train, y_train, X_test, y_test = self.preprocess_data()            

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify= y_train)        
        input_dim = X_train.shape[1]
        
        model = Sequential()
        model.add(LSTM(64, input_shape = (1,X_train.shape[2]), return_sequences=True, unroll=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(LSTM(32,unroll=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(self.attack_types, activation ='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
        early_stop = EarlyStopping(monitor='val_loss', mode = 'auto', 
                           patience = 5 , verbose =1, restore_best_weights=True)
        model.fit(x = X_train, y = y_train, epochs = 100, batch_size = 256, verbose = 0,
                  callbacks = [early_stop], validation_data = (X_val,y_val))
        
        return model, X_test, y_test
    
    def run_model(self, iterations):
             
        results = pd.DataFrame(columns=["acc","rec","prec","far","f1","Train Time","Test Time"])    
        self.preprocess_data()
        
        for i in range(iterations):
            
            # Timing the time taken for training and testing model
            print(f"Iteration {i} \n Model training in progress.")
            training_time = timeit(lambda: self.train_model(), number = 1)
            
            model, X_test, y_test = self.train_model()
            print(f" Model testing in progress.")
            testing_time = timeit(lambda: model.predict(X_test), number = 1)          
            y_pred = np.argmax(model.predict(X_test), axis=-1)
            
            acc = metrics.accuracy_score(y_test,y_pred)
            prec = metrics.precision_score(y_test,y_pred,average='weighted')
            rec = metrics.recall_score(y_test,y_pred,average='weighted')
            f1 = metrics.f1_score(y_test,y_pred,average='weighted')
            
            # Determine false alarm rate using the confusion matrix
            confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)
            FAR = FP/(FP+TN)
            
            model_results = {'acc':acc, 'rec':rec, 'prec':prec, 'far':FAR.mean(), 'f1':f1,
                             'Train Time':training_time,'Test Time':testing_time}
            results = results.append(model_results, ignore_index= True)
            print(f" Model {i} accuracy:", acc, "\n") 
        self.acc = results.mean()[0], results.std()[0]
        self.rec = results.mean()[1], results.std()[1]
        self.prec = results.mean()[2], results.std()[2]
        self.far = results.mean()[3], results.std()[3]  
        self.f1 = results.mean()[4], results.std()[4]
        self.train_time = results.mean()[5], results.std()[5]
        self.test_time = results.mean()[6], results.std()[6]