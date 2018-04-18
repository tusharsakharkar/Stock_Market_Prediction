import pandas_datareader.data as web
from datetime import datetime
start = datetime(2015, 2, 9)
end = datetime(2017, 5, 24)
df=web.DataReader(name='GOOGL',data_source='morningstar',start=start,end=end)
def inc_dec(c, o):
    if c > o:
        value = "Increase"
    elif c < o:
        value = "Decrease"
    else:
        value = "Equal"
    return value

df["Status"] = [inc_dec(c, o) for c, o in zip(df.Close, df.Open)]
df["Middle"] = (df.Open + df.Close) / 2
df["Height"] = abs(df.Close - df.Open)
#print(df)
print(df.index.levels[1,])