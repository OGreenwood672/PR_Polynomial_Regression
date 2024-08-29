from pandas import read_csv
import numpy as np

def load_mtn(path):
    df = read_csv(path)
    df = df[df['pressure'] != 0]
    df['acc'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)

    return df





if __name__ == "__main__" :
    res = load_mtn("./mtn/sensor_data.mtn")
    print(res)