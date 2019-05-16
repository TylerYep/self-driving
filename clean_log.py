import pandas as pd
import csv
import const

def main():
    df = pd.read_csv(const.DRIVING_LOG_PATH)
    df.columns = ['ct_path', 'lt_path', 'rt_path', 'steer', 'throttle', 'brake', 'speed', 'high_level_control']
    df['ct_path'] = df['ct_path'].apply(lambda x: x[x.find('IMG/'):])
    df['lt_path'] = df['lt_path'].apply(lambda x: x[x.find('IMG/'):])
    df['rt_path'] = df['rt_path'].apply(lambda x: x[x.find('IMG/'):])
    df.to_csv(const.DRIVING_LOG_PATH, index=False)

if __name__ == '__main__':
    main()