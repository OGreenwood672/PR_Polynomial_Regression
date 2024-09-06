import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import configparser

# Initialize the parser
config = configparser.ConfigParser()

# Read the config file
config.read('config.ini')

ACCELERATION_SMOOTHING = config['processing'].getfloat('acceleration_smoothing')
BAROMETER_SMOOTHING = config['processing'].getfloat('barometer_smoothing')
LOWER_ACC_THRESHOLD = config['processing'].getfloat('lower_acc_threshold')
UPPER_ACC_THRESHOLD = config['processing'].getfloat('upper_acc_threshold')
C_LIM = config['processing'].getint('c_lim')

def load_mtn(path):
    df = pd.read_csv(path)
    # translate_step(df, 'pressure')
    if 'pressure' in df.columns:
        translate_forward(df, 'pressure', 190)

        df = df[df['pressure'] != 0.0]
        df = df.reset_index(drop=True)

    df[['acc_x', 'acc_y', 'acc_z']] = df[['acc_x', 'acc_y', 'acc_z']] * 9.81

    if df['timestamps'].iloc[0] > 10000:
        df['timestamps'] = (df['timestamps'] - df['timestamps'].iloc[0]) / 1000
    
    df['aAcc'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)

    return df


def translate_forward(df, col, steps):
    for i in range(len(df) - steps):
        df.at[i, col] = df[col].iloc[i + steps]
    
    for i in range(len(df) - steps, len(df)):
        df.at[i, col] = 0


def save_df(df, path):
    df.to_csv(path, index=False)


def drift_correction(df, bound):
    rolling_sum = 0
    num_of_pnts = 0
    for acc in df['aAcc']:
        if acc == df['aAcc'].iloc[0] or abs(rolling_sum / num_of_pnts - acc) < bound:
            rolling_sum += acc
            num_of_pnts += 1
    df['aAcc'] -= rolling_sum / num_of_pnts


def smooth(df, col, SMOOTHING):

    smoothed_values = []
    prev = None
    for row in df[col]:

        if prev is None:
            smoothed_value = row
        else:
            smoothed_value = (prev + SMOOTHING * row) / (1 + SMOOTHING)
        
        prev = smoothed_value
        smoothed_values.append(smoothed_value)

    df[col] = smoothed_values


def correct_vel_at_stationary(df, col, effect):
    assert "aStages" in df.columns, "aStages needs to be in the dataframe"
    assert col in df.columns, f"{col} needs to be in the dataframe"
    for i in range(len(df)):
        if df['aStages'].iloc[i] == 0:
            if df[col].iloc[i] > effect:
                df.loc[i:, [col]] = df.loc[i:, [col]] - effect
            elif df[col].iloc[i] < - effect:
                df.loc[i:, [col]] = df.loc[i:, [col]] + effect
            else:
                df.at[i, col] = 0

def differentiate(df, u, v, new_name):
    assert u in df.columns, f"DataFrame must include {u}"
    assert v in df.columns, f"DataFrame must include {v}"
    df[new_name] = 0.0

    for i in range(1, len(df)):
        du = df[u].iloc[i] - df[u].iloc[i - 1]
        dv = df[v].iloc[i] - df[v].iloc[i - 1]

        if dv != 0:
            df.at[i, new_name] = du / dv


def integrate(df, u, v, new_name):
    assert u in df.columns, f"DataFrame must include {u}"
    assert v in df.columns, f"DataFrame must include {v}"
    df[new_name] = 0.0

    for i in range(1, len(df)):
        avg_u = (df[u].iloc[i] + df[u].iloc[i - 1]) / 2
        dv = df[v].iloc[i] - df[v].iloc[i - 1]
        
        df.at[i, new_name] = df[new_name].iloc[i - 1] + avg_u * dv


def add_stages(df, LB, UB, col, prefix):
    max_acc = max(df[col])
    new_col = prefix + "Stages"
    a_lt = LB * max_acc
    a_ut = UB * max_acc

    df[new_col] = 0
    C = 0
    Clim = 20
    for i in range(len(df)):
        acc = df[col].iloc[i]
        curr_stage = df[new_col].iloc[i]

        if abs(acc) > a_ut: C += 1
        else: C = 0

        if abs(acc) < a_lt:
            if curr_stage == 1:
                df.loc[i:, [new_col]] = 2
            elif curr_stage == 3:
                df.loc[i:, [new_col]] = 0
        
        if C >= Clim and curr_stage in [0, 2]:
            j = i
            while abs(df[col].iloc[j]) > a_lt:
                j -= 1
            df.loc[j:, [new_col]] = df[new_col].iloc[i] + 1


def linear_offset(df, label, offset, start, stop):
    dt = df['timestamps'].iloc[stop] - df['timestamps'].iloc[start]
    for i in range(start, stop):
        df.at[i, label] = df[label].iloc[i] - offset * ((df['timestamps'].iloc[i] - df['timestamps'].iloc[start]) / dt)


def chart(col1, col2=None):
    fig, ax1 = plt.subplots()

    labels = {
        'aAcc': 'Acceleration (m/s^2)',
        'aJerk': 'Jerk (m/s^3)',
        'aHeight': 'Height (m)',
        'aVelocity': 'Velocity (m/s)'
    }
    col1_label = labels[col1]

    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel(col1_label, color='tab:blue')
    ax1.plot(df['timestamps'], df[col1], color='tab:blue', label=col1)
    ax1.tick_params(axis='y', labelcolor='tab:blue')   
    plt.axhline(y=0, color='r', linestyle='--', label='y=0')

    if col2 != None:
        ax2 = ax1.twinx()
        ax2.set_ylabel(col2, color='tab:red')
        ax2.plot(df['timestamps'], df[col2], color='tab:red', label=col2)
        ax2.tick_params(axis='y', labelcolor='tab:red')

        minimum = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
        maximum = max(ax1.get_ylim()[1], ax2.get_ylim()[1])
        ax1.set_ylim(minimum, maximum)
        ax2.set_ylim(minimum, maximum)

        plt.title(f'{col1_label} and {col2} vs. Time')
    
    else:
        plt.title(f'{col1_label} vs. Time') 

    fig.tight_layout()
    plt.show()


directory = "./mtn"
for f in os.listdir(directory):
    df = load_mtn(os.path.join(directory, f))
    smooth(df, 'aAcc', ACCELERATION_SMOOTHING)
    drift_correction(df, bound=0.02)
    add_stages(df, LOWER_ACC_THRESHOLD, UPPER_ACC_THRESHOLD, 'aAcc', 'a')
    differentiate(df, 'aAcc', 'timestamps', 'aJerk')
    integrate(df, 'aAcc', 'timestamps', 'aVelocity')
    add_stages(df, 0.2, 0.6, 'aVelocity', 'av')
    correct_vel_at_stationary(df, 'aVelocity', 0.0005)
    integrate(df, 'aVelocity', 'timestamps', 'aHeight')


    if (df['aStages'] == 2).sum() > 200:
        save_dir = "results-a"
    else:
        save_dir = "results-b"
        
    save_df(df, f"./{save_dir}/" + str(hash(os.path.join(directory, f))))


chart('aHeight')
chart('aVelocity')
chart('aAcc')
chart('aJerk')