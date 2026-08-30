import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder,KBinsDiscretizer
from sklearn.metrics import roc_auc_score
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat,get_feature_names


train_set=pd.read_csv('./train_set.csv')
val_set=pd.read_csv('./val_set.csv')
item_set=pd.read_csv('./item_set.csv')

train_set['label']=(train_set['behavior_type']=='alipay').astype(int)
val_set['label']=(val_set['behavior_type']=='alipay').astype(int)

train_set=pd.merge(train_set,item_set,on='item_id',how='left')
val_set=pd.merge(val_set,item_set,on='item_id',how='left')
train_set['category_id']=train_set['category_id'].fillna('-1')
val_set['category_id']=val_set['category_id'].fillna('-1')
train_set['price']=train_set['price'].fillna(0)
val_set['price']=val_set['price'].fillna(0)

user_click_counts=train_set.groupby('user_id').size().reset_index(name='user_click_counts')
item_click_counts=train_set.groupby('item_id').size().reset_index(name='item_click_counts')

train_set=pd.merge(train_set,user_click_counts,on='user_id',how='left')
val_set=pd.merge(val_set,user_click_counts,on='user_id',how='left')
train_set=pd.merge(train_set,item_click_counts,on='item_id',how='left')
val_set=pd.merge(val_set,item_click_counts,on='item_id',how='left')

train_set['user_click_counts']=train_set['user_click_counts'].fillna(0)
val_set['user_click_counts']=val_set['user_click_counts'].fillna(0)
train_set['item_click_counts']=train_set['item_click_counts'].fillna(0)
val_set['item_click_counts']=val_set['item_click_counts'].fillna(0)

bucket_feature=['user_click_counts','item_click_counts','price']
kbd=KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
train_set[bucket_feature]=kbd.fit_transform(train_set[bucket_feature])
val_set[bucket_feature]=kbd.transform(val_set[bucket_feature])

sparse_feature=['user_id','item_id','category_id','price','user_click_counts','item_click_counts']
combined_data=pd.concat([train_set,val_set],axis=0)
for feat in sparse_feature:
    lbe=LabelEncoder()
    combined_data[feat]=lbe.fit_transform(combined_data[feat].astype(str))
len_train=len(train_set)
train_set=combined_data.iloc[:len_train].copy()
val_set=combined_data.iloc[len_train:].copy()

fixlen_feature_columns=[
    SparseFeat(feat,vocabulary_size=combined_data[feat].nunique(),embedding_dim=4)
    for feat in sparse_feature
]
feature_names=get_feature_names(fixlen_feature_columns)
train_model_input={name:train_set[name].values for name in feature_names}
val_model_input={name:val_set[name].values for name in feature_names}
y_train=train_set['label'].values
y_val=val_set['label'].values
device='cuda'
model=DeepFM(
    linear_feature_columns=fixlen_feature_columns,
    dnn_feature_columns=fixlen_feature_columns,
    task='binary',
    device=device
)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['auc'])
history=model.fit(train_model_input,
                  y_train,
                  batch_size=512,
                  epochs=4,
                  verbose=2,
                  validation_split=0)
y_pred_prob=model.predict(val_model_input)
auc_score=roc_auc_score(y_val,y_pred_prob)
print(f"AUC Score: {auc_score}")