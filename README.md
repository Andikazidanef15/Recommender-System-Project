# RECOMMENDER SYSTEM PROJECT
Create menu's recommendation based on user's preference and user's customer segmentation by using Neural Collaborative Filtering (NCF) [He et al., 2017](https://arxiv.org/abs/1708.05031) and K-Means Clustering
. Due to sensitive information in the dataset, the data used in the recommender/cluster model is not stored in this data folder. 

# API SETUP
Follow these steps to run the API code for generating sales and menus recommendation for particular user:
1. Insert the data (csv format) in the data folder
2. Install all required packages
```
pip install -r requirements.txt
```
3. Run main.py to start the flask API
```
python main.py
```
4. Input User ID listed in the data and run this code,
```
curl -X POST http://127.0.0.1:80/recms -H 'Content-Type: application/json' -d '{"user_id":"XXXXXXXX"}'
```
5. You can get the sales and menus recommendation in following format,
```
{'sales_id':['AAA','BBB',...], 'menu_id': ['CCC', 'DDD', ...]}
```

# TRAINING AND EVALUATE MODEL
To train and evaluate NCF model run these steps:
1. Insert the data (csv format) in the data folder
2. Install all required packages
```
pip install -r requirements.txt
```
3. Run train.py and specify model parameters with arguments, for example if you want to train the model from beginning:
```
python train.py --device cpu/gpu --num_factors 10 --num_hiddens '[10,10,10]' --num_neg 8 --epochs 50 --batch_size 16 --weight_decay 1e-5 --lr 0.01
```
If you want to use pretrained model, you can use this
```
python train.py --checkpoint lightning_logs/
```
