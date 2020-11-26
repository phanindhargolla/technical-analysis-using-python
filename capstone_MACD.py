import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import talib as ta
import datetime as dt
from pandas.plotting import lag_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates


def app():

    price_data = pd.read_csv('.\\allOHLCV.csv')
    price_data_dc = price_data.loc[:, ['Symbol', 'date', 'close']]

    def local_css(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    local_css("style.css")

    option = st.sidebar.selectbox(
        'Select the company',
        price_data['Symbol'].unique())

    data1 = price_data_dc.loc[price_data['Symbol']
                              == option, ['close', 'date']]

    st_date = st.sidebar.date_input("Start Date", dt.date(2018, 1, 1))
    ed_date = st.sidebar.date_input("End Date", dt.date(2020, 9, 30))

    data1 = data1.loc[((data1.date >= str(st_date)) & (
        data1.date <= str(ed_date))), ['close', 'date']]

    macd = st.sidebar.checkbox("MACD")
    if macd:
        st.title("Moving Average Convergence and Divergence")
        # st.line_chart(data1['close'])
        #st.line_chart(price_data_dc.loc[price_data['Symbol'] == 'ALK',['close','date']])
        data1['date'] = pd.to_datetime(data1['date'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        ax1.plot(data1['date'], data1['close'], label=(
            'close price of ' + option), color='blue')
        ax1.legend(loc='upper left')
        exp1 = data1['close'].ewm(span=12, adjust=False).mean()
        exp2 = data1['close'].ewm(span=26, adjust=False).mean()
        macd = exp1-exp2
        exp3 = macd.ewm(span=9, adjust=False).mean()
        ax2.plot(data1['date'], macd, label='MACD', color='#FF0000')
        ax2.plot(data1['date'], exp3, label='Signal Line', color='#00FF00')
        ax2.legend(loc='upper left')
        plt.show()
        st.pyplot(fig)

    bollinger = st.sidebar.checkbox('Bollinger Band')

    days = st.sidebar.slider('No of Days for Moving Average', 8, 25, 20)

    if bollinger:
        st.title("Bollinger Bands")
        data1['close_sma_20'] = data1['close'].rolling(
            days, min_periods=1).mean()
        data1['close_sma_20_std'] = data1['close'].rolling(
            days, min_periods=1).std()
        data1['upper_band'] = data1['close_sma_20'] + \
            data1['close_sma_20_std'] * 2
        data1['lower_band'] = data1['close_sma_20'] - \
            data1['close_sma_20_std'] * 2
        data1['date'] = pd.to_datetime(data1['date'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        ax.plot(data1['date'], data1['close'], label=(
            'close price of ' + option), color='blue', lw=2)
        ax.plot(data1['date'], data1['close_sma_20'], label=(
            str(days) + ' day Moving Average'), color='red', lw=2)
        ax.plot(data1['date'], data1['upper_band'],
                label='upper band', color='green')
        ax.plot(data1['date'], data1['lower_band'],
                label='lower band', color='brown')
        ax.fill_between(data1['date'], data1['lower_band'],
                        data1['upper_band'], color='gray')
        ax.legend(loc='upper left')
        plt.show()
        st.pyplot(fig)

    moving_average = st.sidebar.checkbox('Moving Average')

    a, b = st.sidebar.slider("Select Moving Average Range:",
                             10, 200, (50, 200), step=10)

    sma_ema = st.sidebar.selectbox("Average Type", ['SMA', 'EMA'])

    if moving_average:
        if sma_ema == 'SMA':
            st.title('Simple Moving Average')
            data1['date'] = pd.to_datetime(data1['date'])
            data1['MA1'] = data1['close'].rolling(a, min_periods=1).mean()
            data1['MA2'] = data1['close'].rolling(b, min_periods=1).mean()
            st.set_option('deprecation.showPyplotGlobalUse', False)
            fig, ax = plt.subplots(1, 1, figsize=(15, 12))
            ax.plot(data1['date'], data1['close'],
                    label=('close price of ' + option), color='blue', lw=2)
            ax.plot(data1['date'], data1['MA1'], label=(
                str(a) + ' day Simple Moving Average'), color='red')
            ax.plot(data1['date'], data1['MA2'], label=(
                str(b) + ' day Simple Moving Average'), color='green')
            ax.legend(loc='upper left')
            plt.show()
            st.pyplot(fig)
        elif sma_ema == 'EMA':
            st.title('Exponential Moving Average')
            data1['date'] = pd.to_datetime(data1['date'])
            st.set_option('deprecation.showPyplotGlobalUse', False)
            exp1 = data1['close'].ewm(span=a, adjust=False).mean()
            exp2 = data1['close'].ewm(span=b, adjust=False).mean()
            fig, ax = plt.subplots(1, 1, figsize=(15, 12))
            ax.plot(data1['date'], data1['close'],
                    label=('close price of ', option), color='blue', lw=2)
            ax.plot(data1['date'], exp1, label=(
                str(a) + ' day Exponential Moving Average'), color='red')
            ax.plot(data1['date'], exp2, label=(
                str(b) + ' day Exponential Moving Average'), color='green')
            ax.legend(loc='upper left')
            plt.show()
            st.pyplot(fig)

    rsi = st.sidebar.checkbox('RSI')

    if rsi:
        st.title('Relative Strength Index')
        data1['date'] = pd.to_datetime(data1['date'])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        ax1.plot(data1['date'], data1['close'],
                 label=('close price of ' + option))
        data1['RSI'] = ta.RSI(data1['close'], 14)
        ax2.plot(data1['date'], data1['RSI'], label='RSI')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper left')
        plt.show()
        st.pyplot(fig)

    brain_data = pd.read_csv(
        '.\\updatedBrainBuzz.csv')
    brain_data['CALCULATION_DATE'] = pd.to_datetime(
        brain_data['CALCULATION_DATE'])

    brain_data_opt = brain_data.loc[brain_data['PRIMARY_EXCHANGE_TICKER'] == option, [
        'CALCULATION_DATE', 'SENTIMENT_SCORE_30DAY']]

    br_data = brain_data_opt.loc[(brain_data.CALCULATION_DATE.dt.year >= 2018) & (
        brain_data.CALCULATION_DATE.dt.month < 10), ['CALCULATION_DATE', 'SENTIMENT_SCORE_30DAY']]

    br_data = br_data.loc[((br_data.CALCULATION_DATE >= str(st_date)) & (
        br_data.CALCULATION_DATE <= str(ed_date))), ['CALCULATION_DATE', 'SENTIMENT_SCORE_30DAY']]

    sentiment = st.sidebar.checkbox("Sentiment Scores Correlation")

    if sentiment:
        st.title("Sentiment vs Stock Price")
        data1['date'] = pd.to_datetime(data1['date'])
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        ax1.plot(data1['date'], data1['close'],
                 label=('close price of ' + option))
        ax2.plot(br_data['CALCULATION_DATE'], br_data['SENTIMENT_SCORE_30DAY'])
        plt.show()
        st.pyplot(fig)

    predictive = st.sidebar.checkbox('Predictive Model')

    if predictive:
        st.title("Predictive Analysis")
        train_data, test_data = data1[0: int(
            len(data1) * 0.8)], data1[int(len(data1) * 0.8):]

        def smape_kun(y_true, y_pred):
            return np.mean((np.abs(y_pred - y_true) * 200 / (np.abs(y_pred) + np.abs(y_true))))

        train_ar = train_data['close'].values
        test_ar = test_data['close'].values

        history = [x for x in train_ar]
        predictions = []
        for t in range(len(test_ar)):
            model = ARIMA(history, order=(5, 1, 0))
            model_fit = model.fit(disp=0)
            output = model_fit.forecast()
            yhat = output[0][0]
            predictions.append(yhat)
            obs = test_ar[t]
            history.append(obs)

        error = mean_squared_error(test_ar, predictions)

        data1['date'] = pd.to_datetime(data1['date'])
        fig, ax = plt.subplots(figsize=(12, 7))
        plt.plot(data1['close'], 'green', color='blue', label='Training Data')
        plt.plot(test_data.index, predictions, color='green', marker='o', linestyle='dashed',
                 label='Predicted Price')
        plt.plot(test_data.index,
                 test_data['close'], color='red', label='Actual Price')
        plt.title(option + ' Prices Prediction')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        #date_form = mdates.DateFormatter("%Y-%m-%d")
        # ax.xaxis.set_major_formatter(date_form)
        # plt.xticks('')
        plt.show()
        st.pyplot(fig)


if __name__ == '__main__':
    app()
