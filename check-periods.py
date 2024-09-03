import pandas as pd
import os
from matplotlib import pyplot as plt

directory = './lr_data'

for filename in sorted(os.listdir(directory)):
    if filename.endswith('.csv'):
        file_path = os.path.join(directory, filename)
        file_path = "./results/-6939339446156862666"
        
        df = pd.read_csv(file_path)
        
        plt.figure(figsize=(10, 6))
        plt.plot(df['timestamps'], df['aAcc'], label='Acceleration')
        
        # Set the title to the filename (without the directory path)
        plt.title(f'{filename}')
        
        # Label the axes
        plt.xlabel('Timestamps')
        plt.ylabel('Acceleration (aAcc)')
        
        # Display the legend
        plt.legend()
        
        # Show the plot
        plt.show()
        
        # Pause the execution until the user closes the plot
        input("Press Enter to continue to the next file...")
        break