import pandas as pd
import csv

def main():
    csv_driving_data = 'data/driving_log.csv'
    df = pd.read_csv(csv_driving_data)
    df.columns = ['ct_path', 'lt_path', 'rt_path', 'steer', 'throttle', 'brake', 'speed']
    df['ct_path'] = df['ct_path'].apply(lambda x: x[x.find('IMG/'):])
    df['lt_path'] = df['lt_path'].apply(lambda x: x[x.find('IMG/'):])
    df['rt_path'] = df['rt_path'].apply(lambda x: x[x.find('IMG/'):])
    df.to_csv(csv_driving_data, index=False)

if __name__ == '__main__':
    main()