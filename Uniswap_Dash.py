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
def load_unique_pairs():
    query = """
    SELECT DISTINCT CONCAT(buy, '-', sell) AS pair
    FROM `tristerotrading.uniswap.v3_trades`
    WHERE buy IN ('USDC', 'USDT')
       OR sell IN ('USDC', 'USDT')
    """
    
    query_job = client.query(query)
    results = query_job.result()

    # Convert query results to a DataFrame
    df = query_job.to_dataframe()
    return df['pair'].tolist()


@st.cache_data
def load_data(pair=None):
    # SQL query with dynamic pair filtering
    query = f"""
    WITH trades_with_cumulative AS (
      SELECT
        buy,
        sell,
        -- Correct trade volume: whichever side involves USDC or USDT
        CASE
          WHEN buy IN ('USDC', 'USDT') THEN quantity_buy
          WHEN sell IN ('USDC', 'USDT') THEN quantity_sell
          ELSE 0  -- Adding the ELSE clause to handle cases where neither side is USDC/USDT
        END AS trade_volume,
        -- Cumulative volume using the correct trade volume
        SUM(CASE
          WHEN buy IN ('USDC', 'USDT') THEN quantity_buy
          WHEN sell IN ('USDC', 'USDT') THEN quantity_sell
          ELSE 0
        END) OVER (ORDER BY
          CASE
            WHEN buy IN ('USDC', 'USDT') THEN quantity_buy
            WHEN sell IN ('USDC', 'USDT') THEN quantity_sell
            ELSE 0
          END) AS cumulative_volume,
        -- Total volume for the pair
        SUM(CASE
          WHEN buy IN ('USDC', 'USDT') THEN quantity_buy
          WHEN sell IN ('USDC', 'USDT') THEN quantity_sell
          ELSE 0
        END) OVER () AS total_volume
      FROM
        `tristerotrading.uniswap.v3_trades`
      WHERE
        (buy IN ('USDC', 'USDT') OR sell IN ('USDC', 'USDT'))
        {f"AND CONCAT(buy, '-', sell) = '{pair}'" if pair else ""}
    )
    SELECT 
      buy,
      sell,
      trade_volume,
      cumulative_volume,
      total_volume,
      cumulative_volume / total_volume AS cumulative_percentage
    FROM trades_with_cumulative
    ORDER BY trade_volume;
    """

    # Execute the query
    query_job = client.query(query)
    results = query_job.result()

    # Convert the query result to a DataFrame
    df = query_job.to_dataframe()

    return df


#def load_data():
#    query = """
#    SELECT *
#    FROM `tristerotrading.uniswap.v3_trades`
#    WHERE buy IN ('USDC', 'USDT')
#       OR sell IN ('USDC', 'USDT')
#    LIMIT 50000;

 #   """

 #   query_job = client.query(query)

 #   results = query_job.result()

 #   df = query_job.to_dataframe()

 #   df['volume'] = df.apply(
 #       lambda row: row['quantity_buy'] if 'USDT' in row['buy'] or 'USDC' in row['buy'] else (
 #                   row['quantity_sell'] if 'USDT' in row['sell'] or 'USDC' in row['sell'] else 0),
 #       axis=1
 #   )

 #   return df


# Load the data
#df = load_data()
#data = df
pairs = load_unique_pairs()
pairs.append('Total')
'''
Uniswap Data Dashboard

Browse Uniswap data by pair from a large collection of data.

'''

selected_pairs = st.multiselect('Which pairs would you like to view?', pairs, ['USDC-ENS', 'USDT-UNI', 'USDT-USDC', 'Total'])

''
''
''

cvf_combined_data = pd.DataFrame()

file_path = "bquxjob_1398a98e_1939337331a.csv"
# Fetch and combine data for the selected pairs
for pair in selected_pairs:
    if pair != 'Total':  # Exclude 'Total' from the loop
        df = load_data(pair)
        if not df.empty:
            # Calculate the CVF curve
            df['cumulative_percentage'] = df['cumulative_volume'] / df['total_volume']
            df['log_volume'] = np.log10(df['trade_volume'])

            # Add a new column to label the pair
            df['pair'] = pair
            
            # Append the data to the combined DataFrame
            cvf_combined_data = pd.concat([cvf_combined_data, df], ignore_index=True)
    else:
        # If 'Total' is selected, calculate the CVF for all pairs combined
        loaded_df = pd.read_csv(file_path)

        aligned_df = pd.DataFrame({
            'buy': np.nan,  # Assume 'buy' is not available in the CSV
            'sell': np.nan,  # Assume 'sell' is not available in the CSV
            'trade_volume': np.nan,  # Set default values for missing columns
            'cumulative_volume': np.nan,
            'total_volume': np.nan,
            'cumulative_percentage': loaded_df['percentage_of_total_volume'],  # Use existing data
            'log_volume': loaded_df['log_volume'],  # Use the existing log_volume from the CSV
            'pair': 'Total'  # Label all rows as 'Total'
        })

        cvf_combined_data = pd.concat([cvf_combined_data, aligned_df], ignore_index=True)


if not cvf_combined_data.empty:
    
    #st.write("Maximum cumulative percentage for each pair:")
    #st.write(cvf_combined_data.groupby('pair')['cumulative_percentage'].max())

    st.header("Cumulative Volume Curves for Select Pairs on Uniswap")

    ''
    
    # Initialize an empty list to hold the sampled data for each pair
    sampled_dfs = []

    # Get the unique pairs that the user has selected
    selected_pairs = cvf_combined_data['pair'].unique()

    # Loop over the selected pairs and sample 1000 points for each
    for pair in selected_pairs:
        # Filter the data for the current pair
        pair_data = cvf_combined_data[cvf_combined_data['pair'] == pair]

        # Sort by 'log_volume' to ensure it's in the correct order
        pair_data = pair_data.sort_values(by='log_volume')

         # Normalize log_volume to start at 0
        min_log_volume = pair_data['log_volume'].min()
        pair_data['log_volume'] -= min_log_volume

        # Ensure there's a point at log_volume = 0
        if 0 not in pair_data['log_volume'].values:
            # Add a row with log_volume = 0
            interpolated_row = {
                'log_volume': 0,
                'cumulative_percentage': 0,  # Start cumulative percentage at 0
                'cumulative_volume': 0,      # Start cumulative volume at 0
                'total_volume': pair_data['total_volume'].iloc[0],  # Keep total volume consistent
                'pair': pair
            }
            pair_data = pd.concat([pd.DataFrame([interpolated_row]), pair_data], ignore_index=True)
            pair_data = pair_data.sort_values(by='log_volume')
        
        # Select 1000 evenly spaced points
        num_points = 5000
        if len(pair_data) > num_points:
            indices = np.linspace(0, len(pair_data) - 1, num_points, dtype=int)
            sampled_pair_data = pair_data.iloc[indices]
        else:
            sampled_pair_data = pair_data  # If there are fewer than 1000 points, use them all

        # Append the sampled data for the pair to the list
        sampled_dfs.append(sampled_pair_data)

    # Combine all the sampled data for each pair into one DataFrame
    sampled_combined_data = pd.concat(sampled_dfs, ignore_index=True)

    # Pivot the data to have 'log_volume' as the index and pairs as columns
    #chart_data = sampled_combined_data.pivot_table(
    #    index='log_volume',
    #    columns='pair',
    #    values='cumulative_percentage',
    #    aggfunc='max'  # To handle duplicate log_volume values
    #)

    # Plot the combined CVF data with 1000 evenly spaced points for each pair
    
    st.line_chart(sampled_combined_data, x='log_volume', y='cumulative_percentage', color='pair')
else:
    st.warning("No data available to plot.")


''
''
st.header('Total Volume For Each Pair')

for pair in selected_pairs:

    if pair == 'Total':

        Total_Volume = float(np.pow(loaded_df['log_volume'][9999],10))
        st.metric(
            label = 'Total Volume of All Trades'
            value = Total_Volume
        )
        
# Plot all selected CVF curves on the same graph
#if not cvf_combined_data.empty:
#    st.write("CVF Curves for Selected Pairs:")

    # Pivot the data to have pairs as columns and log_volume as index
#    chart_data = cvf_combined_data.pivot_table(
#        index='log_volume',
#        columns='pair',
#        values='cumulative_percentage',
#        aggfunc='max'  # To handle duplicate log_volume values
#    )

    # Plot the combined CVF data
#    st.line_chart(chart_data, use_container_width=True)
#else:
#    st.warning("No data available to plot.")
#st.write("Data loaded:", df.shape)
#st.dataframe(df.head())

#df = load_data()
# Create a new DataFrame to store the trades for each pair
#trades_by_pair = []

# Group by the unique pairs (buy, sell)
#for (buy, sell), group in df.groupby(['buy', 'sell']):
    # For each pair, we store the trades in a dictionary or DataFrame
 #   pair_df = group.copy()  # Copy the subset corresponding to the pair
  #  pair_df['pair'] = f"{buy}-{sell}"  # Add a new column for the pair identifier
  #  trades_by_pair.append(pair_df)  # Append this DataFrame to the list

#df['pair'] = df.apply(lambda row: f"{row['buy']}-{row['sell']}", axis=1)

# Combine all the DataFrames for each pair into a single DataFrame
#result_df = pd.concat(trades_by_pair, ignore_index=True)

#pairs = result_df['pair'].unique()

#pairs = list(df['pair'].unique()) + ['Total']

#if not len(pairs):
    
#    st.warning("Select at least one pair")
    

#selected_pairs = st.multiselect('Which pairs would you like to view?', pairs, ['USDC-ENS', 'USDT-UNI', 'USDT-USDC', 'Total'])

# Remove 'Total' from selected_pairs for filtered_pairs calculation
#selected_pairs_without_total = [pair for pair in selected_pairs if pair != 'Total']

#filtered_pairs = result_df[(result_df['pair'].isin(selected_pairs))]

#filtered_pairs = df[(df['pair'].isin(selected_pairs_without_total))]

# Generate CVF points
#def calculate_cvf(data):
#    data = data.sort_values('volume')
#    data['cumulative_volume'] = data['volume'].cumsum()
#    data['cumulative_percentage'] = (data['cumulative_volume'] / data['cumulative_volume'].iloc[-1])
#    return data[['volume', 'cumulative_percentage', 'pair']]

#cvf_data = pd.concat([calculate_cvf(filtered_pairs[filtered_pairs['pair'] == pair]) for pair in selected_pairs])

#individual_cvf_data = pd.concat(
#    [calculate_cvf(filtered_pairs[filtered_pairs['pair'] == pair]) for pair in selected_pairs_without_total]
#)

# Compute CVF for 'Total' (all pairs) if 'Total' is selected
#if 'Total' in selected_pairs:
    # CVF for all data in result_df (no filtering by pair)
#    total_cvf_data = calculate_cvf(df)
#    total_cvf_data['pair'] = 'Total'  # Label the total data
#    cvf_data = pd.concat([individual_cvf_data, total_cvf_data])  # Combine with individual pairs
#else:
#    cvf_data = individual_cvf_data  # Only include individual pairs

#if 'Total' in selected_pairs:
#    selected_pairs_without_total = [pair for pair in selected_pairs if pair != 'Total']
#    filtered_pairs = df[df['pair'].isin(selected_pairs_without_total)]
#else:
#    filtered_pairs = df[df['pair'].isin(selected_pairs)]

#st.write("Filtered pairs after selection:", filtered_pairs.shape)
#st.dataframe(filtered_pairs.head())  # Check the first few rows after filtering
#st.write("Columns in filtered pairs:", filtered_pairs.columns.tolist())
#for p in filtst.write("Columns in filtered pairs:", filtered_pairs.columns.tolist())ered_pairs['pair'].unique():
    
    
 #   np.array(result_df.loc[result_df['pair'] ==p]['volume'].tolist())
 
# Log-scale adjustment for x-axis
#cvf_data['log_volume'] = np.log10(cvf_data['volume'])


# Plot with Streamlit
#st.line_chart(data=filtered_pairs, x='log_volume', y='cumulative_percentage', color='pair')

