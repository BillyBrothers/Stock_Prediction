import pandas as pd
import plotly.graph_objects as go
import streamlit as st

def interactive_candlesticks(df: pd.DataFrame, ticker: str):
        """
        Plots an interactive candlestick chart using Plotly and Streamlit.

        Parameters:
        df (pd.DataFrame): DataFrame containing stock data with columns: ['Open', 'High', 'Low', 'Close']
        ticker (str): Stock ticker symbol for chart title


        Returns:
        None: Displays the plot in a Streamlit app.
        """
        fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
        )])

        fig.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
        )

        st.plotly_chart(fig, use_container_width=True)