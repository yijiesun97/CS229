import pandas as pd

## read in data
data = pd.read_csv("company_industry_relation.csv", sep="|")
data = data[['company_id', 'industry_sector', 'is_primary']]
data = data[data['is_primary'] == 'y']
data.drop('is_primary', axis=1, inplace=True)

industry_name = pd.DataFrame({'industry_sector': pd.Series([16, 17, 18, 19, 20, 21, 22]),
                              'industry': pd.Series(['B2B', 'B2C', 'Energy', 'Finance', 'Healthcare', 'IT', 'Materials'])})
data = pd.merge(data, industry_name, on='industry_sector', how='left')
data.drop(['industry_sector'], axis=1, inplace=True)

## save data
data.to_csv("output_industry.csv", index=False)