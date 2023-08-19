# Import Package
import pandas as pd
import argparse
import pytorch_lightning as pl
from utils.load_data import encode_data, split_data
from NeuMF import NCF

# Create Argument Parser
def parse_args():
    parser = argparse.ArgumentParser(description="Run NCF")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--device', nargs='?', default='cpu',
                         help = 'Use CPU/GPU for Training')
    parser.add_argument('--num_factors', type = int, default = 10,
                        help='Number of Embedding Dimension')
    parser.add_argument('--num_hiddens', nargs='?', default = '[10, 10, 10]',
                        help='Number of neurons in each MLP layer')
    parser.add_argument('--num_neg', type = int, default = 8,
                        help='Number of negative random samples in the data')
    parser.add_argument('--epochs', type = int, default = 25,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type = int, default = 16,
                        help='Number of batch size')
    parser.add_argument('--weight_decay', type = float, default = 1e-5,
                        help='L2 Regularizer in Optimizer')
    parser.add_argument('--lr', type = float, default = 0.01,
                        help='Optimizer Learning Rate')
    parser.add_argument('--checkpoint', nargs = '?', default = '',
                        help="Use checkpoint if checkpoint path != '' ")
    return parser.parse_args()

if __name__ == '__main__':
    # Get all args
    args = parse_args()
    data_path = args.path + 'data.csv'
    device = args.device
    num_factors = args.num_factors
    num_hiddens = eval(args.num_hiddens)
    num_neg = args.num_neg
    epochs = args.epochs
    batch_size = args.batch_size
    weight_decay = args.weight_decay
    lr = args.lr
    checkpoint = args.checkpoint

    # Read Data
    df = pd.read_csv(data_path)

    # Get Encoded Data
    df_model = encode_data(df)

    # Split Data
    train, test = split_data(df_model)

    # Get num users
    num_users = len(df_model['user_id'].unique())
    num_items = len(df_model['menu_id'].unique())
    all_menu_ids = df_model['menu_id'].unique()

    # Initiate NCF Model
    # If checkpoint path not specified train model from beginning of epoch
    if checkpoint == '':
        model = NCF(num_factors = num_factors,
                    num_hiddens = num_hiddens,
                    num_users = num_users,
                    num_items = num_items,
                    num_negatives = num_neg,
                    batch_size = batch_size, 
                    weight_decay = weight_decay, 
                    lr = lr,
                    train_data = train,
                    val_data = test,
                    all_menu_ids = all_menu_ids)
    # If checkpoint path exist train model from checkpoint 
    # note: the model parameters must be the same with previous checkpoint's model
    else:
        model = NCF.load_from_checkpoint(checkpoint)

    # Train Model
    trainer = pl.Trainer(max_epochs = epochs,
                         accelerator = device,
                         logger = True)

    trainer.fit(model)