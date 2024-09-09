import pandas as pd
import os
from matplotlib import pyplot as plt

directory = './lr_data'

for filename in sorted(os.listdir(directory)):
    if filename.endswith('.csv'):# and "-8275661513257109100" in filename:
        file_path = os.path.join(directory, filename)
        
        df = pd.read_csv(file_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamps'], df['aAcc'], label='Acc')
        
        # Set the title to the filename (without the directory path)
        plt.title(f'{filename}')
        
        # Label the axes
        plt.xlabel('Timestamps')
        plt.ylabel('Acceleration')
        
        # Display the legend
        plt.legend()
        
        # Show the plot
        plt.show()
        

"""
-1206614467121466218
-1729228464322526488
-3265401304594168933
"""