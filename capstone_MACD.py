import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Capstone Deliverable")

price_data = pd.read_csv('C:\\Users\\Phani\\Downloads\\allOHLCV.csv')
price_data_dc = price_data.loc[:, ['Symbol', 'date', 'close']]

option = st.sidebar.selectbox(
    'Select the company',
    price_data['Symbol'].unique())

data1 = price_data_dc.loc[price_data['Symbol'] == option, ['close', 'date']]

macd = st.sidebar.checkbox("MACD")
if macd:
    # st.line_chart(data1['close'])
    #st.line_chart(price_data_dc.loc[price_data['Symbol'] == 'ALK',['close','date']])
    data1['date'] = pd.to_datetime(data1['date'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    ax1.plot(data1['date'], data1['close'], label='close', color='blue')
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
    data1['close_sma_20'] = data1['close'].rolling(days, min_periods=1).mean()
    data1['close_sma_20_std'] = data1['close'].rolling(
        days, min_periods=1).std()
    data1['upper_band'] = data1['close_sma_20'] + data1['close_sma_20_std'] * 2
    data1['lower_band'] = data1['close_sma_20'] - data1['close_sma_20_std'] * 2
    data1['date'] = pd.to_datetime(data1['date'])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots(1, 1, figsize=(15, 12))
    ax.plot(data1['date'], data1['close'], label='close', color='blue', lw=2)
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
        data1['date'] = pd.to_datetime(data1['date'])
        data1['MA1'] = data1['close'].rolling(a, min_periods=1).mean()
        data1['MA2'] = data1['close'].rolling(b, min_periods=1).mean()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        ax.plot(data1['date'], data1['close'],
                label='close', color='blue', lw=2)
        ax.plot(data1['date'], data1['MA1'], label=(
            str(a) + ' day Simple Moving Average'), color='red')
        ax.plot(data1['date'], data1['MA2'], label=(
            str(b) + ' day Simple Moving Average'), color='green')
        ax.legend(loc='upper left')
        plt.show()
        st.pyplot(fig)
    elif sma_ema == 'EMA':
        data1['date'] = pd.to_datetime(data1['date'])
        st.set_option('deprecation.showPyplotGlobalUse', False)
        exp1 = data1['close'].ewm(span=a, adjust=False).mean()
        exp2 = data1['close'].ewm(span=b, adjust=False).mean()
        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        ax.plot(data1['date'], data1['close'],
                label='close', color='blue', lw=2)
        ax.plot(data1['date'], exp1, label=(
            str(a) + ' day Exponential Moving Average'), color='red')
        ax.plot(data1['date'], exp2, label=(
            str(b) + ' day Exponential Moving Average'), color='green')
        ax.legend(loc='upper left')
        plt.show()
        st.pyplot(fig)
