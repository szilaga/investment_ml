import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, ARDRegression

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam


class MLLib:
    def get_LinearRegression(self,x_train, y_train):
        '''
        Computes the regression model and predicts the stock price values
        inputs: x train data, y train data,
        output: the linear model
        '''
        # adr = ARDRegression()
        lr = LinearRegression()
        # train model
        lr.fit(x_train, y_train)

        return lr

    def get_MLPRegressor(self,x_train, y_train, max_iter=1000, hls=100):
        '''
        Train neuronal network model to predict stock prices
        :param x_train:
        :param y_train:
        :param max_iter:
        :param hls:
        :return: ml model as object
        '''

        # Normalize Data
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)

        MLP = MLPRegressor(random_state=0, max_iter=max_iter, hidden_layer_sizes=(hls,),
                           activation='identity',
                           learning_rate='adaptive').fit(x_train_scaled, y_train)

        return MLP, scaler

    def get_LSTM(self,x_train, y_train, x_test, y_test):
        '''
        Train LSTM neuronal network model to predict stock prices
        :param x_train:
        :param y_train:
        :return: ml model as object
        '''

        # Convert x_train and y_train to numpy arrays and reshaping
        x_train_data, y_train_data = np.array(x_train), np.array(y_train)
        x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

        # set test data
        x_test_data, y_test_data = np.array(x_test), np.array(y_test)
        x_test_data = np.reshape(x_test_data, (x_test_data.shape[0], x_test_data.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dense(units=50))
        model.add(Dense(units=1))

        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy','mean_squared_error'])
        #model.compile(loss='binary_crossentropy', optimizer='adam')
        model.compile(loss='mse', optimizer='adam', metrics=['mse'])
        #print(model.summary())

        #model.fit(x_train_data, y_train_data, batch_size=100, validation_data=(x_test_data, y_test_data), epochs=10)
        model.fit(x_train_data, y_train_data, batch_size=100, epochs=10, verbose=0)

        # Make prediction
        y_pred = model.predict(x_test_data)

        # Evaluation of the model
        scores = model.evaluate(x_test_data,y_test_data, verbose=1)
        print('R2 Score: ', r2_score(y_test_data, y_pred))
        print('MAE: ', mean_absolute_error(y_test_data, y_pred))
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        return y_pred

    def get_Predictions(self, model, x_test, y_test, modelname):
        '''
        Predict stock price based on linear model
        input: lineare model, x test data, y test data
        output: the predicted values for the test data
        '''
        # make predictions
        y_pred = model.predict(x_test)
        y_MSE = mean_squared_error(y_test, y_pred)
        y_R2 = model.score(x_test, y_test)

        accuracy = (y_pred.round(decimals=1, out=None) == np.array(y_test).round(decimals=1, out=None)).mean()

        print('{} R2: {}'.format(modelname, y_R2))
        print('{} MSE: {}'.format(modelname, y_MSE))
        print("Accuracy:", accuracy)

        return y_pred