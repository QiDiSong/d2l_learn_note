import pandas as pd
df = pd.read_csv('UsersBehavior.csv')
print(df.head())
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date
df.loc[df['behaviorType'] != 'pv', 'behaviorType'] = 0
df.loc[df['behaviorType'] == 'pv', 'behaviorType'] = 1
# 按日期分组并计算每日点击量
pv_per_day = df.groupby('date')['behaviorType'].sum()
pv_per_day = pv_per_day.sort_values(ascending=False)
pv_per_day.to_excel('223102张三.xlsx')
