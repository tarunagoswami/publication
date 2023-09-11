import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import os
import plotly.express as px

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split  # Import train_test_split function
from keras.models import Sequential
from keras import metrics
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint


from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
# dataset = pd.read_csv('/Users/taruna/Desktop/publication/Data/impaired-cp/AOF0802R.csv')
# pdf = matplotlib.backends.backend_pdf.PdfPages("output.pdf")

rootdir = "/Users/taruna/Desktop/publication/Data/impaired-cp"
plotDir ='/Users/taruna/Desktop/publication/RNN-plots-impaired-hand'
df_folder = []
target =  "/Users/taruna/Desktop/publication/Data/demographics_cp_reduced.xlsx"
csv_list = []

write_header = True
output_csv = 'combined.csv'

df_target = pd.read_excel(r'/Users/taruna/Desktop/publication/Data/demographics_cp_reduced.xlsx', engine='openpyxl')

print(df_target)
for filename in os.listdir(rootdir):
    f = os.path.join(rootdir, filename)
    if f.endswith('.csv'):
        path = os.path.basename(os.path.normpath(f))
        print('path')
        findIndx = path.index('.')
        patient = path[:findIndx]
        print(patient)
        dataset = pd.read_csv(f)

        # dataset = dataset.iloc[:-5]
        # dataset['C'] = dfObject.C.str.replace(r"[a-zA-Z]",'')

        print(dataset)
        features =  ['LH.POS.x','LH.POS.y', 'LH.POS.z','LH.ROT.x', 'LH.ROT.y', 'LH.ROT.z', 'TH.POS.x', 'TH.POS.y','TH.POS.z','TH.ROT.x','TH.ROT.y','TH.ROT.z']

        X = dataset[features]

        print(np.shape(features))
        # X= np.array(X)
        print(X)
        train_scaled = np.reshape(X, (X.shape[0], X.shape[1]))
        from sklearn.preprocessing import MinMaxScaler  # scaling the attributes 0-1

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(X)
        train_scaled
        mean_vec = np.mean(train_scaled, axis=0)
        cov_mat = (train_scaled - mean_vec).T.dot((train_scaled - mean_vec)) / (train_scaled.shape[0] - 1)
        print('NumPy covariance matrix: \n%s' % np.cov(train_scaled.T))
        plt.figure(figsize=(8, 8))
        sns.heatmap(cov_mat, vmax=1, square=True, annot=True, cmap='cubehelix')
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        print('Eigenvectors \n%s' % eig_vecs)
        print('\nEigenvalues \n%s' % eig_vals)
        plt.title('Correlation between different features')
        print('Covariance matrix \n%s' % cov_mat)

        # dataset['features'].unique()
        
        from sklearn.decomposition import PCA

        # pca = PCA(n_components=1, svd_solver='arpack')
        # pca.fit(train_scaled)
        # PCA(n_components=1, svd_solver='arpack')
        # print(pca.explained_variance_ratio_)
        # print(pca.singular_values_)

        pca = PCA(n_components=3)
        components = pca.fit_transform(train_scaled)
        print('----pca----')
        print(pca)
        print(pca.explained_variance_ratio_)
        print('important features')

        total_var = pca.explained_variance_ratio_.sum() * 100
        print(abs(pca.components_))
      
        for i in df_target['Name'] :
            if(i == patient) :
    # Split dataset into training set and test set
                X = pca.components_
                scaler = MinMaxScaler()
                scaler.fit(X)
                X = scaler.transform(X)
                target_data   = df_target.loc[df_target['Name'] == i]
                df_new = pd.DataFrame(target_data, columns=['MACS'])
                df_array = np.array([df_new])
                
                
                y = np.array([df_array, df_array, df_array])
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=1)

                y = np.array(y)
                y = y.reshape(-1, 1)
                
                
                print(X.shape)
                print(y.shape)  
                X_test, y_test = np.array(X_test), np.array(y_test)
                print('X_test')
                print(X_test)
                print('X_train')
                print(X_train)
                # # Initialize RNN:
                # model = Sequential()

                # # Adding the first RNN layer and some Dropout regularization
                # model.add(SimpleRNN(units=1000, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
                # model.add(Dropout(0.5))

                # # Adding the second RNN layer and some Dropout regularization
                # model.add(SimpleRNN(units=1000, activation='relu', return_sequences=True))
                # model.add(Dropout(0.5))

            
                # # # Adding the third RNN layer and some Dropout regularization
                # model.add(SimpleRNN(units=1000, activation='relu', return_sequences=True))
                # model.add(Dropout(0.5))


                # # Adding the fourth RNN layer and some Dropout regularization
                # model.add(SimpleRNN(units=1000))
                # model.add(Dropout(0.2))

              

                #  # Adding the output layer
                # model.add(Dense(units=1))
                # model.add(Dropout(0.5))
              


                # # Compile the RNN
                # model.compile(optimizer='adam', loss='mean_squared_error')

                def lstm_layer (hidden1) :
    
                    model = Sequential()
                    
                    # add input layer
                    model.add(Input(shape = (X_train.shape[1], 1)))
                    
                    # add rnn layer
                    model.add(LSTM(hidden1, activation = 'tanh', return_sequences = False))
                    model.add(BatchNormalization())
                    model.add(Dropout(0.2))
                    
                    # add output layer
                    model.add(Dense(1, activation = 'linear'))
                    
                    model.compile(loss = "mean_squared_error", optimizer = 'adam')
                    
                    return model

                model = lstm_layer(256)
                # model.summary()             
                

                print(model.summary())
                print('model compiled')

                X_train, y_train = np.array(X_train), np.array(y_train)
                X_test, y_test = np.array(X_test), np.array(y_test)
                X_train = np.reshape(X_train, (X_train.shape[0], -1, 1))
                X_test = np.reshape(X_test,  (-1, 12, X_test.shape[0]))

                # checkp = ModelCheckpoint('./bit_model_lstm.h5', monitor = 'val_loss', save_best_only = True, verbose = 1)     
                
                print(X_train.shape)
                print(X_test.shape)
                # history = model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)
                # history = model.fit(X_train, y_train, epochs=10, batch_size=32,validation_split=0.2)
                y_test = y_test.reshape(-1, 1)
                y_test = np.array(y_test)
                # Predict = model.predict(X_test)
                # train_predict = model.predict(X_train)
                # test_predict = model.predict(X_test)
                # accuracy = model.evaluate(X_test, y_test, verbose=1)
                model.fit(X_train, y_train, batch_size = 32, epochs = 10, validation_data = (X_test, y_test))
                # print(f'Test results - Loss: {accuracy[0]} - Accuracy: {accuracy[1]*100}%')

                # mse = mean_squared_error(y_test, Predict)
                # print(f"The mean squared error for {filename}: " + str(mse))
                pred = model.predict(X_test)
                print(pred.shape)
                pred = pred.reshape(-1)
                print('MSE : ' + str(mean_squared_error(y_test, pred)))
                plt.figure(figsize = (20,7))
                plt.plot(y_test[-1:1])
                plt.plot(pred[-1:1])
                plt.xlabel('Time')
                plt.ylabel('MACS')
                plt.title('MACs (using LSTM)')
                plt.legend(['Actual', 'Predicted'])
                plt.show()

               
                # plt.plot(history.history['accuracy'])
                # plt.plot(history.history['val_accuracy'])
                # plt.title("Plot of accuracy vs epoch for train and test dataset")
                # plt.ylabel('accuracy')
                # plt.xlabel('epoch')
                # plt.show()    
                # x = []
                # plt.plot(Predict, linestyle = 'dotted')
                # for i in range(0,len(y_test)):
                #     x.append(i)
                # plt.scatter(x, y_test)
                # plt.show()


                # def plot_result(y_train, y_test, train_predict, test_predict):
                #     actual = np.append(y_train, y_test)
                #     predictions = np.append(train_predict, test_predict)
                #     rows = len(actual)
                #     plt.figure(figsize=(15, 6), dpi=80)
                #     plt.plot(range(rows), actual)
                #     plt.plot(range(rows), predictions)
                #     plt.axvline(x=len(y_train), color='r')
                #     plt.legend(['Actual', 'Predictions'])
                #     plt.xlabel('Observation number after given time steps')
                #     plt.ylabel('Impared Hand scaled')
                #     plt.title('Actual and Predicted Values')
                #     loc = plotDir + '/' + filename
                #     loc += '.png'
                #     plt.savefig(loc)
                #     plt.close()
                # plot_result(y_train, y_test, train_predict, test_predict)
                # plt.figure(figsize=(10,10))
                # plt.scatter(y_test, Predict, c='crimson')
                # plt.yscale('log')
                # plt.xscale('log')

                # p1 = max(max(Predict), max(y_test))
                # p2 = min(min(Predict), min(y_test))
                # plt.plot([p1, p2], [p1, p2], 'b-')
                # plt.xlabel('True Values', fontsize=15)
                # plt.ylabel('Predictions', fontsize=15)
                # plt.axis('equal')
                # # plt.show()
                # # plt.savefig(plotDir + filename +'.png') 
                # filename = plotDir + '/' + filename
                # filename += '.png'
                # plt.savefig(filename)

            
                # print("Mean squared Error")
                # print(mse)
                