import pandas as pd 
import numpy as np
import json
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from NeuMF import NCF
from utils.load_data import encode_data, split_data, get_user_encoding
from cluster import aggregate_df, cluster

# Read Data
df = pd.read_csv('data/data.csv')

# Get Encoded Data
df_model = encode_data(df)

# Split Data
train, test = split_data(df_model)

# Get User and Menu ID mapping
dict_users_encode, _, _, _ = get_user_encoding(df)

# Include only unique values in menu_id
menus = df.drop_duplicates(subset = ['menu_id'], keep = 'first').reset_index(drop = True)

# Load Model from Checkpoint
model = NCF.load_from_checkpoint('lightning_logs/menus_df/checkpoints/epoch=49-step=504500.ckpt',
                                 map_location ='cpu')

# Create Recommender Function
# Create Menu's Recommender Function
def recommender(user_id):
    """ Create menus recommendation based on User ID by NCF model
        Args:
            user_id (str): User ID
    """
    # Get all unique menus
    menu_list = df_model['menu_id'].unique()
    nunique_menu = len(menu_list)

    # Get User ID map
    user_encode = dict_users_encode[user_id]

    # Get prediction using NCF model
    predicted_labels = np.squeeze(model(torch.tensor([user_encode]*nunique_menu),
                                        torch.tensor(menu_list)).detach().numpy())

    # Get Top 10 Items
    top10_items = [menu_list[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]
    top10_scores = np.sort(predicted_labels)[::-1][0:10]

    menus_subset = menus.iloc[top10_items].reset_index(drop = True).copy()
    menus_subset['confidence'] = top10_scores

    return {'sales_id': menus_subset['sales_id'].tolist(), 'menu_id': menus_subset['menu_id'].tolist()}

# Create Recommendation based on Customer Segmentation
def recommendation_segment(cust_segment, dict_label_name, closest_pt_idx, user_id):
    """ Create menus recommendation based on User's Segment Prediction
        Args:
            cust_segment (pd.DataFrame): DataFrame with User ID as index and Customer Segmentation Label 
            dict_label_name (dict): Dictionary containing mapping from Customer Segmentation Label to custom name
            closest_pt_idx (list): List containing indexes of segment's user representative
            user_id (str): User ID 
    """
    # Get Customer Segment Label
    user_encode = cust_segment.loc[user_id, 'Label']

    # Print Category
    print(f'This customer belongs to {dict_label_name[user_encode]} segment')

    # Get Recommendation for the representative of that customer segment
    menus_rec = recommender(closest_pt_idx[int(user_encode)])

    # Return Result
    return json.dumps(menus_rec)


# Create Aggregated Data
agg_all_scaled = aggregate_df(df)

# Predict Segmentation
agg_all_scaled, closest_pt_idx = cluster(agg_all_scaled)

# Create Flask API
app = Flask(__name__)
CORS(app)

@app.route("/")
def hello_from_root():
    return jsonify(message="Welcome! This API is for retrieving customer's menu recommendation based on")

@app.route("/recms", methods = ["POST"])
def make_rec():
    if request.method == "POST":
        data = request.get_json()
        user_id = data["user_id"]
        #curl -X POST http://127.0.0.1:80/recms -H 'Content-Type: application/json' -d '{"user_id":"72512d30-7126-4b6c-91ce-1a3f00fba4b5"}'
        dict_label_name = {0 : 'Silver Tier Cafe Users',
                            1 : 'Basic Tier Traditional Users',
                            2 : 'Silver Tier Bandung Cafe Users',
                            3 : 'Gold, Black Tier Cafe Users',
                            4 : 'Diamond, Black Tier Traditional Users'}
        api_recommendations = recommendation_segment(agg_all_scaled, dict_label_name, closest_pt_idx, user_id)
        return api_recommendations
  
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=80)





