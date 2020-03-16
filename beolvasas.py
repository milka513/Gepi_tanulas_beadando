import pandas as pd
import numpy as np

listings_data=pd.read_csv('./files/listings.csv')
print(listings_data,' ', listings_data.columns)
calendar_summary_columns=pd.read_csv('./files/calendar_summary/calendar_summary_1.csv').columns
for i in range(1, 13):
    column_name='month {0}'.format(i)
    listings_data[column_name]=pd.Series(np.zeros(len(listings_data['id'])))

for i in range(1, 11):
    filename='./files/calendar_summary/calendar_summary_{0}.csv'.format(i)
    if (i!=1):
        temp=pd.read_csv(filename, names=calendar_summary_columns)
    else:
        temp = pd.read_csv(filename)
    temp['date']=pd.to_datetime(temp['date'])
    for month in range(1, 13):
        column_name = 'month {0}'.format(i)
        for id in listings_data[['id']].values:
            #print(month, id)
            listings_data.loc[listings_data['id']==id[0], column_name]+= len(temp[(temp['date'].dt.month==month)
              & (temp['available']=='t') &
                       (temp['listing_id']==id[0])].values)
print(listings_data)
listings_data.to_csv('summed_data.csv')