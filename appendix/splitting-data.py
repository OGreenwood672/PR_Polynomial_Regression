import os
import pandas as pd
import numpy as np



def get_periods(result_file, buffer):
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

        if not len(period) or i >= len(df): continue

        for j in range(max(0, i - len(period) - 1), max(i - len(period) - buffer, previous_end) - 1, -1):
            period.insert(0, [
                    df['timestamps'].iloc[j],
                    df['aAcc'].iloc[j],
                    df['aJerk'].iloc[j],
                    df['aVelocity'].iloc[j],
                    df['aHeight'].iloc[j]
                ])

        sliced_df = df.iloc[i + 1:]
        next_stage = sliced_df[sliced_df['aStages'].apply(lambda x: x in [1, 3])].index
        next_stage = len(df) - 1 if len(next_stage) == 0 else next_stage[0]

        for i in range(i, min(int((next_stage - i) / 2 + i), i + buffer)):
            period.append([
                    df['timestamps'].iloc[i],
                    df['aAcc'].iloc[i],
                    df['aJerk'].iloc[i],
                    df['aVelocity'].iloc[i],
                    df['aHeight'].iloc[i]
                ])

        previous_end = i

        period_df = pd.DataFrame(period, columns=['timestamps', 'aAcc', 'aJerk', 'aVelocity', 'aHeight'])

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
        period_df['aAcc'] = np.where(period_df['aAcc'].abs() < 0.01, 0, period_df['aAcc'])

        period_df['aVelocity'] = period_df['aVelocity'] - period_df['aVelocity'].iloc[0]
        period_df['aVelocity'] = abs(period_df['aVelocity'])
        period_df['aVelocity'] = np.where(period_df['aVelocity'] < 0.01, 0, period_df['aVelocity'])

        period_df['aHeight'] = period_df['aHeight'] - period_df['aHeight'].iloc[0]
        period_df['aHeight'] = abs(period_df['aHeight'])
        period_df['aHeight'] = np.where(period_df['aHeight'] < 0.01, 0, period_df['aHeight'])
        
        period_dfs.append(period_df)

    return period_dfs


directory = os.listdir("./results-a")
for csv_file in directory:
    periods = get_periods(csv_file, 100)
    for index, period in enumerate(periods):
        period.to_csv("lr_data/" + csv_file + "-period-" + str(index) + ".csv", index=False)