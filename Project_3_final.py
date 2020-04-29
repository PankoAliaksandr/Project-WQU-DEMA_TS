# Libraries

from pandas_datareader import data as pdr
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA


# Class implementation
class EMA:

    # Constructor
    def __init__(self):
        self.__index_data = pd.DataFrame()
        self.__window_size = 200
        self.__trends = pd.DataFrame(columns=['Range', 'Coefficients'])

        self.__download_data()

    # Getters
    def get_index_data(self):
        return self.__index_data

    def get_trend_data(self):
        return self.__trends

    def __download_data(self):
        # Determine the first and the last days of the last 10 years period
        end_date = datetime.date.today()
        start_date = datetime.date(end_date.year - 15, end_date.month,
                                   end_date.day)
        # Index data
        self.__index_data = pdr.get_data_yahoo('^DJI', start_date, end_date)
#        self.__index_data = self.__index_data['Adj Close']
        self.__index_data.dropna(inplace=True)

        # Create a new column for deviations
        self.__index_data['Mid Price'] = (self.__index_data['Low'] +
                                          self.__index_data['High']) / 2

    def __calculate_DEMA(self):
        self.__index_data['EMA'] = self.__index_data['Close'].ewm(
                ignore_na=False, span=200, adjust=False).mean()

    def __calculate_mid_price_deviation(self):
        self.__index_data['Deviation'] = ((self.__index_data['Mid Price'] -
                                          self.__index_data['EMA']) /
                                          self.__index_data['EMA'])

    def __find_ups_and_downs(self):
        # If market is Boolish mark the period with TRUE value else FALSE
        self.__index_data['Market Type'] = self.__index_data['Deviation'] > 0

    def __find_all_trends(self):

        k = 0 # Trend number
        i = 0 
        min_trend_num = 3

        n = len(self.__index_data)
        while(i < (n - 1)):
            start = i
            j = i
            counter = 0
            while(self.__index_data['Market Type'][j] ==
                  self.__index_data['Market Type'][j+1]):
                if (j == (n - 2)):
                    j = j + 1
                    counter = counter + 1
                    break
                else:
                    j = j + 1
                    counter = counter + 1

            # End of period: make a trend line
            i = j + 1
            # Range will cut the last day otherwise
            end = j + 1

            if(counter >= min_trend_num):
                # Get all prices during the trend
                prices = self.__index_data.iloc[range(start, end)]['Close']
                # Compute the trend line coefficients
                coefficients = np.polyfit(range(start, end), prices, 1)
                self.__trends.loc[k] = [range(start, end), coefficients]
                k = k + 1

    def __calculate_delta_slopes(self):
        coef = self.__trends['Coefficients']
        all_slopes = np.zeros(shape=(len(coef), 1))
        for i in range(len(all_slopes)):
            all_slopes[i] = coef[i][0]

        delta_slopes = np.zeros(shape=(len(all_slopes), 1))
        for i in range(1, len(all_slopes)):
            delta_slopes[i-1] = np.abs(all_slopes[i] - all_slopes[i-1])

        return delta_slopes

    def __calculate_trend_length(self):
        trend_range = self.__trends['Range']
        length = np.zeros(shape=(len(trend_range), 1))
        for i in range(len(trend_range)):
            length[i] = len(trend_range[i])

        return length

    def __do_regression(self):
        delta_slopes = self.__calculate_delta_slopes()
        length = self.__calculate_trend_length()

        explanatory_variable = sm.add_constant(length)
        model = sm.OLS(delta_slopes, explanatory_variable)
        results = model.fit()
        print(results.summary())

        # Visualization
        plt.plot(delta_slopes, label='Actual')
        plt.plot(results.fittedvalues, color='red',
                 label='Predicted')
        plt.title("How trend length predicts changes?")
        plt.legend()
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()

    def __forcast_last(self):
        coef = self.__trends['Coefficients']
        all_slopes = np.zeros(shape=(len(coef), 1))
        for i in range(len(all_slopes)-1):
            all_slopes[i] = coef[i][0]

        model = ARIMA(all_slopes, order=(1, 0, 0))
        ARIMA100 = model.fit()
        print ARIMA100.summary()

        predicted_values_ARIMA100 = ARIMA100.fittedvalues
        plt.title("Predicted vs Actual")
        plt.plot(predicted_values_ARIMA100,
                 label="Predicted by ARIMA(1,0,0)")
        plt.plot(all_slopes, label="Actual")
        plt.legend()
        plt.show()

    def __forecase_last_sm(self):
        coef = self.__trends['Coefficients']
        all_slopes = np.zeros(shape=(len(coef), 1))
        for i in range(len(all_slopes)):
            all_slopes[i] = coef[i][0]

        slopes = all_slopes[1:(len(all_slopes)-1)]
        slopes_regressor = all_slopes[0:(len(all_slopes)-2)]
        model = sm.OLS(slopes, slopes_regressor)
        results = model.fit()
        print(results.summary())

        plt.plot(slopes, label='Actual')
        plt.plot(results.fittedvalues, color='red',
                 label='Predicted')
        plt.title("Lag prediction?")
        plt.legend()
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()
    # Visualization

    def __plot_mid_price(self):
        plt.plot(self.__index_data['Mid Price'],
                 label="DJIA mid price")
        plt.plot(self.__index_data['EMA'],
                 label="DJIA EMA")
        plt.title("DJIA mid price over last 15 years")
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.show()

    def __plot_deviations(self):
        plt.plot(self.__index_data['Deviation'],
                 label="DJIA mid price deviations")
        plt.title("DJIA mid price deviations over last 15 years")
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.show()

    def __plot_market_type(self):
        plt.plot(self.__index_data['Market Type'],
                 label="Market Type")
        plt.title("Market Type: Boolish VS Bearish")
        plt.legend()
        plt.gcf().autofmt_xdate()
        plt.show()

    def __plot_trends(self):
        plt.figure(figsize=(10, 7))
        plt.plot(self.__index_data['Mid Price'],
                 label="DJIA mid price")
        for i in range(len(self.__trends)):
            # Create a polynomial function of one argument
            fit_fun = np.poly1d(self.__trends['Coefficients'][i])

            # Calculate approximational function values
            func_value = lambda x: fit_fun(x)

            if(self.__trends['Coefficients'][i][0] > 0):
                clr = 'black'
            else:
                clr = 'r'
            plt.plot(self.__index_data.index[self.__trends['Range'][i]].values, func_value(
                    self.__trends['Range'][i]), color=clr, linewidth=2)

        plt.title("Trends")
        plt.gcf().autofmt_xdate()
        plt.show()

    def main(self):
        self.__calculate_DEMA()
        self.__calculate_mid_price_deviation()
        self.__plot_mid_price()
        self.__plot_deviations()
        self.__find_ups_and_downs()
        self.__plot_market_type()
        self.__find_all_trends()
        self.__plot_trends()
        self.__do_regression()
        self.__forcast_last()
        self.__forecase_last_sm()


EMA = EMA()
EMA.main()
data = EMA.get_index_data()
trend_data = EMA.get_trend_data()

