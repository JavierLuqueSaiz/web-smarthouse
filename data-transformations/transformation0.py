import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('HomeC.csv', delimiter=',', header=0,)  # Ajusta el delimitador y el tipo de datos según tu archivo

start_time = datetime(2016, 1, 1, 0, 0, 0)  # 1 de enero de 2016 a las 5:00 AM

df['time'] = start_time + pd.to_timedelta(df.index, unit='m')

df.head()

df.to_csv('data1.csv', index=False,sep=",")
