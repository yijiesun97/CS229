import pandas as pd
import numpy as np
from functools import reduce

## read in data
companies = pd.read_csv("output_companies.csv")
deals = pd.read_csv("output_deals.csv")
industry = pd.read_csv("output_industry.csv")
people = pd.read_csv("output_people.csv")
locations = pd.read_csv("locations.csv")[['entity_id', 'state', 'country']]
locations.rename(columns={'entity_id': 'company_id'}, inplace=True)

## merge data
data = reduce(lambda x, y: pd.merge(x, y, on='company_id', how='outer'), [companies, deals, industry, people, locations])

## filter data
data.dropna(inplace=True)
data = data[data.year_founded >= 1998]
data = data[data.country == 'United States']
data.drop(['country'], axis=1, inplace=True)
print(data.shape)
print(data.head(20))

## One-hot encoding
year_founded = pd.get_dummies(data['year_founded'], drop_first=True)
updated_financing = pd.get_dummies(data['updated_financing'], drop_first=True)
gender = pd.get_dummies(data['gender'], drop_first=True)
state = pd.get_dummies(data['state'], drop_first=True)
stock_type = pd.get_dummies(data['updated_stock_type'], drop_first=True)
sector = pd.get_dummies(data['industry'], drop_first=True)

data.drop(['year_founded', 'updated_financing', 'gender', 'state', 'updated_stock_type', 'industry'], axis=1, inplace=True)
data = pd.concat([data, year_founded, updated_financing, gender, stock_type, sector, state], axis=1)

cols = list(data.columns)
new = cols[2]
cols[2] = cols[4]
cols[4] = new

data = data[cols]
print(data.head())
print(data.shape)

## save data
data.to_csv("Processed.csv", index=False)