import pandas as pd
import numpy as np

## read in the data
data = pd.read_csv("deals.csv")

data_sum = data.groupby('company_id', as_index=False)['number_of_tranches', 'number_of_sellers'].sum()
data['stock_type'] = data.groupby('company_id')['stock_type'].ffill()
data.drop_duplicates(subset='company_id', keep='last', inplace=True)

values = {'total_debt': 0, 'raised_to_date': 0}
data.fillna(value=values, method=None, axis=None, inplace=True, limit=None, downcast=None)

conditions = [
    (data['stock_type'] == 'Common'),
    (data['stock_type'] == 'Convertible Preferred'),
    (data['stock_type'] == 'Participating Preferred'),
    (data['stock_type'] == 'Options') | (data['stock_type'] == 'Warrants'),
    (data['stock_type'] == 'Preferred')]
choices = ['Common', 'Convertible Preferred', 'Participating Preferred', 'Options', 'Preferred']
data['updated_stock_type'] = np.select(conditions, choices, default='Others')

data = data[['company_id', 'deal_number', 'total_debt', 'raised_to_date', 'updated_stock_type']]
data = pd.merge(data, data_sum, on='company_id', how='outer')

## save data
data.to_csv("output_deals.csv", index=False)