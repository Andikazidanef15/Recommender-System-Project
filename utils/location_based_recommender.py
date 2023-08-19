# Import Package
import pandas as pd
import pytorch_lightning as pl
import numpy as np
import torch
from utils.load_data import encode_data, split_data, get_user_encoding

# Calculate Euclidean Distance
def euclidean_distance(lat_shop, lat_user, long_shop, long_user):
    return np.sqrt((lat_shop - lat_user)**2 + (long_shop - long_user)**2)

# Create Location Based Menu's Recommender Function
def recommender_location_based(df, model, user_id, district_lat_long):
    """ Create menus recommendation based on User ID and Outlet's Location by NCF model
        Args:
            df (pd.DataFrame): Original DataFrame
            model (NeuMF.NCF): Pretrained NCF model
            user_id (str): User ID
            district_lat_long (dict): Dictionary containing latitude and longitude coordinates from each district with format:
            {'district1': [lat, long]}
    """
    # Subset menu_type,menu_category,menu_category_details, menu_name,
    # Include only unique values
    menus = df.drop_duplicates(subset = ['menu_id'], keep = 'first').reset_index(drop = True)

    # Get the latitude and longitude coordinate frome each district
    menus['lat'] = menus['district'].apply(lambda val: district_lat_long[val][0])
    menus['long'] = menus['district'].apply(lambda val: district_lat_long[val][1])

    # Get User and Menu ID mapping
    dict_users_encode, _, _, _ = get_user_encoding(df)

    # Get Encoded Data
    df_model = encode_data(df)

    # Calculate District Mode from each users
    users_district = df_model.groupby(['user_id'])['district'].agg(lambda x: pd.Series.mode(x)[0])

    # Get all unique menus
    menu_list = df_model['menu_id'].unique()
    nunique_menu = len(menu_list)  

    menus['lat'] = menus['district'].apply(lambda val: district_lat_long[val][0])
    menus['long'] = menus['district'].apply(lambda val: district_lat_long[val][1])

    # Get Encoded User ID
    user_encode = dict_users_encode[user_id]

    # Get prediction using NCF model
    predicted_labels = np.squeeze(model(torch.tensor([user_encode]*nunique_menu),
                                        torch.tensor(menu_list)).detach().numpy())

    # Get All predicted Items and probability score
    items = [menu_list[i] for i in np.argsort(predicted_labels)[::-1].tolist()]
    scores = np.sort(predicted_labels)[::-1]

    # Get User's location
    user_loc = users_district[user_encode]
    user_lat = district_lat_long[user_loc][0]
    user_long = district_lat_long[user_loc][1]

    # Calculate euclidean distance between all shops
    menus_subset = menus.iloc[items].copy()
    menus_subset['confidence'] = scores
    menus_subset['distance'] = menus_subset[['lat','long']].apply(lambda outlet: euclidean_distance(outlet.lat,
                                                                                                    user_lat,
                                                                                                    outlet.long,
                                                                                                    user_long), axis = 1)

    # Sort menus_subset
    menus_subset = menus_subset.sort_values(by = ['distance','confidence'], ascending = [True, False])

    # Get Top 10 Recommendations
    return menus_subset[['sales_id', 'menu_id']].head(10)