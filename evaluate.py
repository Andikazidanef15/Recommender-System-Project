# Import Package
import pandas as pd 
import numpy as np
import argparse
import torch

from NeuMF import NCF
from utils.load_data import encode_data, split_data

# Create Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run NCF")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--pretrained', nargs = '?', default = 'lightning_logs/menus_df/checkpoints/epoch=49-step=504500.ckpt',
                        help="Use pretrained or last checkpoint model")
    return parser.parse_args()

if __name__ == '__main__':
    # Get all args
    args = parse_args()
    data_path = args.path + 'data.csv'
    pretrained = args.pretrained

    # Read Data
    df = pd.read_csv(data_path)

    # Get Encoded Data
    df_model = encode_data(df)

    # Split Data
    train, test = split_data(df_model)

    # Get num users
    all_menu_ids = df_model['menu_id'].unique()

    # Load Model from Checkpoint
    model = NCF.load_from_checkpoint(pretrained,
                                     map_location ='cpu')

    # User-item pairs for testing
    test_user_item_set = set(zip(test['user_id'], test['menu_id']))

    # Dict of all items that are interacted with by each user
    user_interacted_items = df_model.groupby('user_id')['menu_id'].apply(list).to_dict()

    # Get number of hits
    hits = []

    # Loop for all user item in test dataset
    for (u,i) in test_user_item_set:
        # Get all items that the user interacted with
        interacted_items = user_interacted_items[u]
        # Get all items that the user had not interacted with
        not_interacted_items = set(all_menu_ids) - set(interacted_items)
        # Select 99 items from not_interacted_items
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        # Add with the test items
        test_items = selected_not_interacted + [i]
        # Make predictions
        predicted_labels = np.squeeze(model(torch.tensor([u]*100),
                                              torch.tensor(test_items)).detach().numpy())

        top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]

        if i in top10_items:
          hits.append(1)
        else:
          hits.append(0)

    print("The Hit Ratio @ 10 is {:.4f}".format(np.average(hits)))