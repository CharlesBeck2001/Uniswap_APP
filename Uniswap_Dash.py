#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 12:55:21 2024

@author: charlesbeck
"""
import os
import streamlit as st
import pandas as pd
from google.cloud import bigquery
import numpy as np
import json
import time

st.set_page_config(
    page_title='Uniswap Trades Dashboard'
    )

service_account_info = {
    "private_key": st.secrets["gcp"]["private_key"],
    "private_key_id": st.secrets["gcp"]["private_key_id"],
    "client_email": st.secrets["gcp"]["client_email"],
    "token_uri": st.secrets["gcp"]["token_uri"],
    "project_id": st.secrets["gcp"]["project_id"]
}

# Access the private key from Streamlit secrets
#private_key = st.secrets["gcp"]["private_key"]

# Initialize BigQuery client using the provided private key string
client = bigquery.Client.from_service_account_info(service_account_info)

# Continue with your code as usual
st.write("BigQuery client initialized successfully.")

@st.cache_data
def load_data():
    query = """
    SELECT *
    FROM `tristerotrading.uniswap.v3_trades`
    WHERE buy IN ('USDC', 'USDT')
       OR sell IN ('USDC', 'USDT')
    LIMIT 100000;

    """

    query_job = client.query(query)

    results = query_job.result()

    df = query_job.to_dataframe()

    df['volume'] = df.apply(
        lambda row: row['quantity_buy'] if 'USDT' in row['buy'] or 'USDC' in row['buy'] else (
                    row['quantity_sell'] if 'USDT' in row['sell'] or 'USDC' in row['sell'] else 0),
        axis=1
    )

    return df

# Load the data
df = load_data()

if not data.empty:
    st.write("Data loading complete!")
    st.dataframe(data.head())  # Show the first few rows
else:
    st.warning("No data to display.")

#df = load_data()
# Create a new DataFrame to store the trades for each pair
trades_by_pair = []

# Group by the unique pairs (buy, sell)
#for (buy, sell), group in df.groupby(['buy', 'sell']):
    # For each pair, we store the trades in a dictionary or DataFrame
 #   pair_df = group.copy()  # Copy the subset corresponding to the pair
  #  pair_df['pair'] = f"{buy}-{sell}"  # Add a new column for the pair identifier
  #  trades_by_pair.append(pair_df)  # Append this DataFrame to the list

df['pair'] = df.apply(lambda row: f"{row['buy']}-{row['sell']}", axis=1)

# Combine all the DataFrames for each pair into a single DataFrame
#result_df = pd.concat(trades_by_pair, ignore_index=True)

#pairs = result_df['pair'].unique()

pairs = list(df['pair'].unique()) + ['Total']

if not len(pairs):
    
    st.warning("Select at least one pair")

'''
Uniswap Data Dashboard

Browse Uniswap data by pair from a large collection of data.

'''
    

selected_pairs = st.multiselect('Which pairs would you like to view?', pairs, ['USDC-ENS', 'USDT-UNI', 'USDT-USDC', 'Total'])

''
''
''

# Remove 'Total' from selected_pairs for filtered_pairs calculation
selected_pairs_without_total = [pair for pair in selected_pairs if pair != 'Total']

#filtered_pairs = result_df[(result_df['pair'].isin(selected_pairs))]

filtered_pairs = df[(df['pair'].isin(selected_pairs_without_total))]

# Generate CVF points
def calculate_cvf(data):
    data = data.sort_values('volume')
    data['cumulative_volume'] = data['volume'].cumsum()
    data['cumulative_percentage'] = (data['cumulative_volume'] / data['cumulative_volume'].iloc[-1])
    return data[['volume', 'cumulative_percentage', 'pair']]

#cvf_data = pd.concat([calculate_cvf(filtered_pairs[filtered_pairs['pair'] == pair]) for pair in selected_pairs])

individual_cvf_data = pd.concat(
    [calculate_cvf(filtered_pairs[filtered_pairs['pair'] == pair]) for pair in selected_pairs_without_total]
)

# Compute CVF for 'Total' (all pairs) if 'Total' is selected
if 'Total' in selected_pairs:
    # CVF for all data in result_df (no filtering by pair)
    total_cvf_data = calculate_cvf(df)
    total_cvf_data['pair'] = 'Total'  # Label the total data
    cvf_data = pd.concat([individual_cvf_data, total_cvf_data])  # Combine with individual pairs
else:
    cvf_data = individual_cvf_data  # Only include individual pairs


#for p in filtered_pairs['pair'].unique():
    
    
 #   np.array(result_df.loc[result_df['pair'] ==p]['volume'].tolist())
 
# Log-scale adjustment for x-axis
cvf_data['log_volume'] = np.log10(cvf_data['volume'])

# Plot with Streamlit
st.line_chart(cvf_data, x='log_volume', y='cumulative_percentage', color='pair')

