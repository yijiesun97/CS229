import pandas as pd
import numpy as np

## read in data
data = pd.read_csv("people.csv")[['company_id', 'gender']]

num_founders = data.pivot_table(index=['company_id'], aggfunc='size').to_frame()

data.drop_duplicates(inplace=True)
data.sort_values(by=['company_id', 'gender'], inplace=True)
data.drop_duplicates(subset='company_id', keep='first', inplace=True)

data = pd.merge(data, num_founders, on='company_id', how='outer')
data.rename(columns={0: 'num_partners'}, inplace=True)

print(data.head(50))

## save data
data.to_csv("output_people.csv", index=False)