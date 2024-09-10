import os
import pandas as pd
import numpy as np


def get_periods(result_file, buffer=None):
    with open("./results-a/" + result_file, "r") as f:
        df = pd.read_csv(f)

    period_dfs = []
    i = 0
    previous_end = 0
    while i < len(df):

        period = []
        
        # Collect data until aStages is non-zero and there's at least one period element
        while i < len(df) and (df['aStages'].iloc[i] in [1, 3] or not len(period)):
            if df['aStages'].iloc[i] in [1, 3]:
                period.append([
                    df['timestamps'].iloc[i],
                    df['aAcc'].iloc[i],
                    df['aJerk'].iloc[i],
                    df['aVelocity'].iloc[i],
                    df['aHeight'].iloc[i]
                ])
            i += 1

        # If bad data ignore
        if not len(period) or i >= len(df): continue

        # Split this period element into changing acc sections
        upper_threshold = 0.9 * np.abs(np.array(period)[:, 1]).max()

        curr_top = 0
        for j in range(len(period)):
            if abs(curr_top) > 0:
                if abs(curr_top) < abs(period[j][1]):
                    curr_top = period[j][1]
                else:
                    break
            elif abs(period[j][1]) > upper_threshold:
                curr_top = period[j][1]
        
        period1_df = pd.DataFrame(period[:j], columns=['timestamps', 'aAcc', 'aJerk', 'aVelocity', 'aHeight'])

        period = period[::-1]
        curr_top = 0
        for k in range(len(period)):
            if abs(curr_top) > 0:
                if abs(curr_top) < abs(period[k][1]):
                    curr_top = period[k][1]
                else:
                    break
            elif abs(period[k][1]) > upper_threshold:
                curr_top = period[k][1]
        
        corrected_period = period[:k][::-1]
        period2_df = pd.DataFrame(corrected_period, columns=['timestamps', 'aAcc', 'aJerk', 'aVelocity', 'aHeight'])

        # Calculate buffer sizes
        sliced_df = df.iloc[i + 1:]
        next_stage = sliced_df[sliced_df['aStages'].apply(lambda x: x in [1, 3])].index
        next_stage = len(df) - 1 if len(next_stage) == 0 else next_stage[0]

        mid_index = int(((i - len(period2_df)) + (i - len(period) + len(period1_df))) / 2)
        # Right Buffer 1
        period1_df = pd.concat([period1_df, df.iloc[i - len(period) + len(period1_df):min(i - len(period) + len(period1_df) + buffer, mid_index)]], ignore_index=True)

        # Left Buffer 1
        period1_df = pd.concat([df.iloc[max(previous_end, i - len(period) - buffer):i - len(period)], period1_df], ignore_index=True)

        # Left Buffer 2
        period2_df = pd.concat([df.iloc[max(mid_index, i - len(period2_df) - buffer):i - len(period2_df)], period2_df], ignore_index=True)

        # Right Buffer 2
        period2_df = pd.concat([period2_df, df.iloc[i: i + min(buffer, int((next_stage - i) / 2))]], ignore_index=True)

        previous_end = i;

        for period_df in [period1_df, period2_df]:

            period_df["jerks_direction"] = "up" if period_df['aJerk'].iloc[len(period_df) - 1] - period_df['aJerk'].iloc[0] > 0 else "down"
            period_df["accelerations_direction"] = "up" if period_df['aAcc'].iloc[int((len(period_df) - 1) / 2)] - period_df['aAcc'].iloc[0] > 0 else "down"
            period_df["velocities_direction"] = "up" if period_df['aVelocity'].iloc[len(period_df) - 1] - period_df['aVelocity'].iloc[0] > 0 else "down"
            period_df["heights_direction"] = "up" if period_df['aHeight'].iloc[len(period_df) - 1] - period_df['aHeight'].iloc[0] > 0 else "down"

            period_df["start_jerks"] = period_df['aJerk'].iloc[0]
            period_df["start_accelerations"] = period_df['aAcc'].iloc[0]
            period_df["start_velocities"] = period_df['aVelocity'].iloc[0]
            period_df["start_heights"] = period_df['aHeight'].iloc[0]

            period_df['actual_timestamps'] = period_df['timestamps']
            period_df['timestamps'] = period_df['timestamps'] - period_df['timestamps'].iloc[0]

            period_df['aJerk'] = period_df['aJerk'] - period_df['aJerk'].iloc[0]
            period_df['aJerk'] = abs(period_df['aJerk'])
            period_df['aJerk'] = np.where(period_df['aJerk'] < 0.01, 0, period_df['aJerk'])

            period_df['aAcc'] = period_df['aAcc'] - period_df['aAcc'].iloc[0]
            period_df['aAcc'] = abs(period_df['aAcc'])
            period_df['aAcc'] = np.where(period_df['aAcc'].abs() < 0.01, 0, period_df['aAcc'])

            period_df['aVelocity'] = period_df['aVelocity'] - period_df['aVelocity'].iloc[0]
            period_df['aVelocity'] = abs(period_df['aVelocity'])
            period_df['aVelocity'] = np.where(period_df['aVelocity'] < 0.01, 0, period_df['aVelocity'])

            period_df['aHeight'] = period_df['aHeight'] - period_df['aHeight'].iloc[0]
            period_df['aHeight'] = abs(period_df['aHeight'])
            period_df['aHeight'] = np.where(period_df['aHeight'] < 0.01, 0, period_df['aHeight'])
            
            period_dfs.append(period_df)

    return period_dfs


from matplotlib import pyplot as plt
def chart(df, y1):

    label_y1 = y1.replace("a", "Measured ")

    plt.figure(figsize=(10, 6))

    # plt.plot(df['timestamps'], df[csv_header], label=predicting + " Actual", color='blue')

    # plt.plot(df['timestamps'][:len(df['best_fit'])], df['best_fit'], label=predicting + " Best Fit", color='red')

    plt.plot(df['timestamps'], df[y1], label=label_y1, color='blue')

    # Adding titles and labels
    plt.title(f'{label_y1} vs Timestamps')
    plt.xlabel('Timestamps')
    plt.ylabel(f"{label_y1}")

    # Display the legend
    plt.legend()

    # Show the plot
    plt.show()

directory = os.listdir("./results-a")
for csv_file in directory:
    # csv_file = "-8275661513257109100"
    periods = get_periods(csv_file, 20)
    for index, period in enumerate(periods):
        period.to_csv("lr_data/" + csv_file + "-period-" + str(index) + ".csv", index=False)
    
    # df = pd.read_csv(f"./results-a/{csv_file}")
    # chart(df, 'aAcc')
    # break



