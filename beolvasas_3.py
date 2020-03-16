import  pandas as pd

listings_data=pd.read_csv('summed_data2.csv')

temp=pd.read_csv('./files/listings_summary.csv')
temp=temp[['id','accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type',
     'square_feet', 'amenities', 'review_scores_rating', 'security_deposit']]
listings_data=pd.merge(listings_data, temp, how='left', left_on=['id'], right_on=['id'])

listings_data=listings_data.drop(['name', 'host_id', 'host_name', 'latitude', 'longitude', 'id', 'Unnamed: 0'], axis=1)
print(listings_data.columns)
#listings_data.drop(['name', 'host_id', 'host_name', 'latitude', 'longitude'], axis=1)
listings_data.to_csv('summed_data_vegleges.csv')