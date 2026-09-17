import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / 'data'

train = pd.read_csv(DATA_DIR / 'train.csv')
meal_info = pd.read_csv(DATA_DIR / 'meal_info.csv')
center_info = pd.read_csv(DATA_DIR / 'fulfilment_center_info.csv')

df = train.merge(meal_info, on='meal_id', how='left')
df = df.merge(center_info, on='center_id', how='left')
print('merged shape:', df.shape)

for c in ['center_type', 'category', 'cuisine']:
    df[c] = df[c].astype('category')

df.to_parquet(DATA_DIR / 'merged_clean.parquet', index=False)

# Faz 2 - feature engineering
df = df.sort_values(['center_id', 'meal_id', 'week']).reset_index(drop=True)

df['discount_ratio'] = (df['base_price'] - df['checkout_price']) / df['base_price']

grp = df.groupby(['center_id', 'meal_id'])['num_orders']
df['num_orders_lag_1'] = grp.shift(1)
df['num_orders_lag_4'] = grp.shift(4)
df['num_orders_roll_mean_4'] = (
    df.groupby(['center_id', 'meal_id'])['num_orders']
      .transform(lambda s: s.shift(1).rolling(4).mean())
)

df['weekofyear'] = ((df['week'] - 1) % 52) + 1
df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)

df['log_num_orders'] = np.log1p(df['num_orders'])

encoders = {}
for col in ['center_type', 'category', 'cuisine']:
    le = LabelEncoder()
    df[col + '_enc'] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

with open(DATA_DIR / 'label_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

df.to_parquet(DATA_DIR / 'features.parquet', index=False)
print('Kaydedildi:', df.shape)
print(df.columns.tolist())
