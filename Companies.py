import pandas as pd
import numpy as np

## read in data
data = pd.read_csv("companies.csv")[['company_id', 'company_name', 'business_status', 'financing_status', 'year_founded',
                                     'employee_count', 'competition']]

## recode business status
conditions = [
    (data['business_status'] == 'Profitable'),
    (data['business_status'] == 'Generating Revenue') | (data['business_status'] == "Generating Revenue") |
    (data['business_status'] == "Not Profitable"),
    (data['business_status'] == 'Stealth') | (data['business_status'] == 'Out of Business') |
    (data['business_status'] == 'Bankruptcy: Liquidation') | (data['business_status'] == 'Bankruptcy: Admin/Reorg')]
choices = ['Profitable', 'Generating Revenue', 'Bankruptcy']
data['updated_status'] = np.select(conditions, choices, default='Startup')

## simplified business status
in_business = ['Startup', 'Profitable', 'Generating Revenue']
data['in_business'] = data['updated_status'].apply(lambda x: 'Success' if x in in_business else 'Failure')

## recode financing status
pe_backed = ['Corporate Backed or Acquired', 'Corporation', 'Formerly PE-Backed',
             'Pending Transaction (Debt)', 'Pending Transaction (M&A)', 'Private Debt Financed', 'Private Equity-Backed']
data['updated_financing'] = data['financing_status'].apply(lambda x: "pe_backed" if x in pe_backed else "vc_backed")

## competitiors
data['num_comp'] = data['competition'].fillna(0).apply(lambda x: x if x == 0 else len(str(x).split(',')))

data.drop(['business_status', 'financing_status', 'competition'], axis=1, inplace=True)
print(data.head(50))

## save data
data.to_csv("output_companies.csv", index=False)