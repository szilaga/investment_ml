import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sn

import numpy as np

class Plots:

    def show_CorrelationPlot(self, df):
        '''
        Show correlation matrix of features, to check of they
        are dependend to the stock price
        '''
        # set plot with size
        fig, ax = plt.subplots(figsize=(10, 6))

        # set title
        ax.set_title("Correlation Matrix stock features")

        # set x,y labels
        ax.set_xticklabels(df.columns)
        ax.set_yticklabels(df.columns)

        # set data
        ax = sn.heatmap(df.corr(), annot=True, vmin=0, vmax=1, linewidths=1)

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # workaround hence first and last row is not cut off
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.7, top - 0.7)

        fig.tight_layout()
        plt.show()

    def show_ScatterPlot(self, train, test, pred_1, pred_2, ticker, w, h, model_1, model_2, forecast):
        '''
        Creates a Scatter plot that shows the actual stock price, and the predicted stockprice

        inputs: train dataframe, test dataframe, predicted value (list), ticker ('string'),
        width (int), height (int), name (string)

        output: Plot
        '''

        # Create lines of the training actual, testing actual, prediction
        D1 = go.Scatter(x=train.index, y=train[ticker], name='Train Actual')  # Training actuals
        D2 = go.Scatter(x=test.index, y=test[ticker], name='Test Actual')  # Testing actuals
        D3 = go.Scatter(x=test.index, y=pred_1, name='Prediction {}'.format(model_1))  # Testing predction
        D4 = go.Scatter(x=test.index, y=pred_2, name='Prediction {}'.format(model_2))  # Testing predction
        # D5 = go.Scatter(x=test.index,y=pred_3, name = 'Prediction {}'.format(model_3)) # Testing predction

        # Combine in an object
        line = {'data': [D1, D2, D3, D4],
                'layout': {
                    'xaxis': {'title': 'Date'},
                    'yaxis': {'title': 'Price'},
                    'title': 'Stock prediction' + ' - ' + ticker + ' - ' + str(forecast)
                }}
        # Send object to a figure
        fig = go.Figure(line)

        # Show figure
        fig.show()

    def show_ScatterCorrPlot(self, df1, df2, model):
        '''
        This plot creates a scatter plot which shows the correlation
        between actual and predicted stock prices
        :param df1:
        :param df2:
        :param model:
        :return: none
        '''
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(x=df1, y=df2,
                                 mode='markers',
                                 name='correlation',
                                 opacity=0.95))

        # Add line
        beta, alpha = np.polyfit(df1, df2, 1)
        lrange = np.arange(start=df1.min(), stop=df1.max(), step=0.1)  # set range length
        fig.add_trace(go.Scatter(x=lrange, y=beta * lrange + alpha,
                                 mode='lines',
                                 #name='Line',
                                 opacity=1.0))

        # Edit the layout
        fig.update_layout(title='Correlation results of {}'.format(model),
                          xaxis_title='Actual',
                          yaxis_title='Predict')

        fig.show()

    def show_Histogramm(self, df, columns, title,mean_dr, std_dr):
        '''
        This histogram depicts the number of daily return of a stock
        :param self:
        :param df:
        :param columns: only the daily price column
        :param title:
        :return: none
        '''
        fig = go.Figure()

        for col in columns:
            fig.add_trace(

                trace=go.Histogram(
                    x=df[col],
                    # histnorm='percent',
                    name='control',  # name used in legend and hover labels
                    xbins=dict(  # bins used for histogram
                        start=-1.0,
                        end=1.0,
                        size=0.01
                    ),
                    marker_color='#4169E1',
                    opacity=0.75
                ))

            # Mean line
            y = np.histogram(df[col], bins=[-1, 0, 1])
            y = y[0].max() * 0.8  # get max y value

            mean_shape = {'line': {'color': '#FF4500', 'dash': 'solid', 'width': 1},
                          'type': 'line',
                          'x0': mean_dr,
                          'x1': mean_dr,
                          'y0': -10,
                          'y1': y}

            # Std lines
            std_shape_pos = {'line': {'color': '#B22222', 'dash': 'dot', 'width': 1},
                             'type': 'line',
                             'x0': std_dr,
                             'x1': std_dr,
                             'y0': -10,
                             'y1': y}

            std_shape_neg = {'line': {'color': '#B22222', 'dash': 'dot', 'width': 1},
                             'type': 'line',
                             'x0': -std_dr,
                             'x1': -std_dr,
                             'y0': -10,
                             'y1': y}

            fig.add_shape(mean_shape)
            fig.add_shape(std_shape_pos)
            fig.add_shape(std_shape_neg)

            # Annotations
            mean_annotation = dict(
                x=mean_dr,
                y=y,
                text="Mean = {}".format(mean_dr),
                showarrow=True,
                arrowhead=7,
            )

            std_annotation_pos = dict(
                x=std_dr,
                y=y * 0.8,
                text="Std = {}".format(std_dr),
                showarrow=True,
                arrowhead=7
            )

            std_annotation_neg = dict(
                x=-std_dr,
                y=y * 0.8,
                text="Std = {}".format(std_dr),
                showarrow=True,
                arrowhead=7
            )

            fig.add_annotation(mean_annotation)
            fig.add_annotation(std_annotation_pos)
            fig.add_annotation(std_annotation_neg)

        fig.update_layout(
            title_text='Historgram Price ' + title,  # title of plot
            xaxis_title_text='Value',  # xaxis label
            yaxis_title_text='Count',  # yaxis label
            bargap=0.2,  # gap between bars of adjacent location coordinates
            bargroupgap=0.1  # gap between bars of the same location coordinates
        )

        fig.show()

    def show_BollingerPlot(self, df, columns, title, xaxis_title, yaxis_title):
        '''
        This Plot shows the actual stock price trend with additional indicators
        like bollinger band etc.
        :param df: data
        :param columns: List of columns
        :param title:
        :param xaxis_title:
        :param yaxis_title:
        :return:
        '''

        fig = go.Figure()

        # sets price
        fig.add_trace(go.Scatter(x=df.index, y=df[columns[0]],
                                 mode='lines',  # +markers
                                 name='price',
                                 opacity=0.95))

        # sets upper
        fig.add_trace(go.Scatter(x=df.index, y=df[columns[4]],
                                 opacity=0.75,
                                 name=columns[4],
                                 line=dict(
                                     color='#000000',
                                     dash='dot')
                                 # fill = 'tonexty'
                                 # fillcolor = fillcol(df[column[0]],df[column[2]]) #'rgba(143,188,143, 1.0)'
                                 ))

        # sets lower
        fig.add_trace(go.Scatter(x=df.index, y=df[columns[5]],
                                 opacity=0.75,
                                 name=columns[5],
                                 line=dict(
                                     color='#000000',
                                     dash='dot'),
                                 fill='tonexty',
                                 fillcolor='rgba(147,112,219, 0.2)'
                                 ))

        # sets sma 30
        fig.add_trace(go.Scatter(x=df.index, y=df[columns[2]],
                                 mode='lines',  # +markers
                                 name=columns[2],
                                 opacity=0.65,
                                 line=dict(
                                     color='#FF0000')))

        # sets sma 50
        fig.add_trace(go.Scatter(x=df.index, y=df[columns[3]],
                                 mode='lines',  # +markers
                                 name=columns[3],
                                 opacity=0.65,
                                 line=dict(
                                     color='#FFFF00')))

        # Edit the layout
        fig.update_layout(title=title,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)

        fig.show()



