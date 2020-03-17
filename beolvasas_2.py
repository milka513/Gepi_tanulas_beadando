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
    #print(temp[(temp['date'].dt.month==12) & (temp['available']=='t')])
    for month in range(1, 13):
        print(month)
        column_name = 'month {0}'.format(month)
        df=temp[(temp['date'].dt.month==month) & (temp['available']=='t')]
        new_df=pd.DataFrame()
        #print(df.groupby('listing_id').count().index.values)
        new_df['listing_id']=df.groupby('listing_id').count().index.values
        new_df['freq']=df.groupby('listing_id').count()['date'].values
        #print(df.groupby('listing_id').count()['date'].values)
        #print(pd.merge(new_df, listings_data, how='left', left_on=['listing_id'], right_on=['id']))
        listings_data2=pd.merge(new_df, listings_data, how='right', left_on=['listing_id'], right_on=['id'])

        #print(listings_data2.columns)
        #print(listings_data2['listing_id'].values)
        listings_data2['freq'] = listings_data2['freq'].fillna(0)
        print(column_name, listings_data[column_name].values+listings_data2['freq'].values)
        listings_data[column_name]= listings_data[column_name].values+listings_data2['freq'].values
        if (month==12):
            print(listings_data2['freq'])
            print(listings_data)
       # print(listings_data)
        #print(df)
        #print(month)
        #for id in df[['listing_id']].values:
        #    listings_data.loc[listings_data['id'] == id[0], column_name]+=len(df[(df['listing_id']==id[0])].values)
        #for id in listings_data[['id']].values:
            #print(month, id)
        #    listings_data.loc[listings_data['id']==id[0], column_name]+= len(temp[(temp['date'].dt.month==month)
        #      & (temp['available']=='t') &
        #               (temp['listing_id']==id[0])].values)
#print(listings_data)
#listings_data.drop(['name', 'host_id', 'host_name', 'latitude', 'longitude'], axis=1)
listings_data.to_csv('summed_data2.csv')