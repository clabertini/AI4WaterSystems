# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 14:05:43 2025

@author: Claudia Bertini, Lecturer and Researcher in Hydroinformatics, IHE Delft, Delft, Netherlands
"""

import pandas as pd
import matplotlib.pyplot as plt
import os


'''0) LOAD THE DATA'''

# Define file path and check environment
if 'google.colab' in str(get_ipython()):
    from google.colab import drive
    drive.mount('/content/drive')
    base_path = '/content/drive/My Drive/'
else:
    base_path = os.getcwd()  # Use current directory in Jupyter Notebook

# Ensure file exists before reading
file_name = 'Sieve-orig.xlsx'
file_path = os.path.join(base_path, file_name)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"File not found: {file_path}")

# First load the file with the data from Sieve
df = pd.read_excel(file_path)

'''1) PLOT THE DATA'''
fig, ax1 = plt.subplots(figsize=(10, 5))

# Plot discharge on the primary y-axis
ax1.plot(df['Date'], df['Qt'], color='steelblue', label='Discharge (m³/s)')
ax1.set_xlabel('Time (hours)')
ax1.set_ylabel('Discharge (m³/s)')
ax1.tick_params(axis='y')
ax1.set_ylim(0, 1000)  
ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
ax1.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=10))
plt.xticks(rotation=45)
ax1.set_xlim(df['Date'].loc[0], df['Date'].loc[len(df)-1])  # Set x-axis limits

# Create secondary y-axis for rainfall
ax2 = ax1.twinx()
ax2.bar(df['Date'], df['REt'], color='lightsteelblue',alpha=0.8, label='Rainfall (mm)')
ax2.set_ylabel('Rainfall (mm)')
ax2.tick_params(axis='y')
ax2.set_ylim(0, 20)  # 
ax2.invert_yaxis()  # Invert the secondary y-axis

# Show the plot
fig.tight_layout()
plt.title('Hourly Discharge and Rainfall')
plt.show()
plot_path = os.path.join(base_path, 'Rainfall-Discharge_Plot.png')
plt.savefig(plot_path, dpi=600, format='png')


'''2) CROSS-CORRELATION: CORRELATION BETWEEN TARGET (Qt+1) AND RAINFALL WITH DIFFERENT TIME LAGS'''


# Compute correlation between discharge and past rainfall steps
correlations = {}
for lag in range(1, 10):  # 10 steps back
    df[f'REt_lag{lag}'] = df['REt'].shift(lag)
    correlations[f'{lag}'] = df[['Qt', f'REt_lag{lag}']].corr().iloc[0, 1]

# Print correlation results
for lag, corr in correlations.items():
    print(f'Correlation with {lag}: {corr:.3f}')


# Plot correlation over different lags
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(correlations.keys()), list(correlations.values()), marker='o', linestyle='-', color='lightseagreen')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Correlation coefficient')
ax.set_ylim(-1,1)
ax.set_title('Correlation between Discharge and Lagged Rainfall')
ax.grid(True)
plt.show() 
correlation_plot_path = os.path.join(base_path, 'Rainfall-Discharge_Correlation.png')
plt.savefig(correlation_plot_path, dpi=600, format='png')


'''3) AUTO-CORRELATION: CORRELATION BETWEEN TARGET (Qt+1) AND PAST DISCHARGE'''
qcorrelations = {}
for lag in range(1, 15):  # 10 steps back
    df[f'Qt_lag{lag}'] = df['Qt'].shift(lag)
    qcorrelations[f'{lag}'] = df[['Qt', f'Qt_lag{lag}']].corr().iloc[0, 1]

# Print correlation results
for lag, corr in qcorrelations.items():
    print(f'Correlation with {lag}: {corr:.3f}')

# Plot correlation over different lags
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(list(qcorrelations.keys()), list(qcorrelations.values()), marker='o', linestyle='-', color='goldenrod')
ax.set_xlabel('Lag (hours)')
ax.set_ylabel('Correlation coefficient')
ax.set_ylim(-1,1)
ax.set_xlim(0,13)
ax.set_title('Discharge Autocorrelation')
ax.grid(True)
plt.show()
autocorr_plot_path = os.path.join(base_path, 'Discharge_Autocorrelation.png')
plt.savefig(autocorr_plot_path, dpi=600, format='png')
