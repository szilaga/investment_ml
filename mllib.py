import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.linear_model import LinearRegression, ARDRegression

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense, LSTM


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
        Predict stock price based on neuronal network
        input: lineare model, x test data, y test data
        output: the predicted values for the test data
        '''
        # Scaling data
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train_scaled = scaler.transform(x_train)

        MLP = MLPRegressor(random_state=0, max_iter=max_iter, hidden_layer_sizes=(hls,),
                           activation='identity',
                           learning_rate='adaptive').fit(x_train_scaled, y_train)

        return MLP, scaler

    def get_LSTM(self,x_train, y_train):
        x_train_data1, y_train_data1 = np.array(x_train), np.array(y_train)

        x_train_data2 = np.reshape(x_train_data1, (x_train_data1.shape[0], x_train_data1.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data2.shape[1], 1)))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=25))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train_data2, y_train_data1, batch_size=1, epochs=1)

        return model

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