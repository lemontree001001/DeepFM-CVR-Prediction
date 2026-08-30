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
#数据挖掘，用户点击量以及商品被点击量
user_click_counts=train_set.groupby('user_id').size().reset_index(name='user_click_counts')
item_click_counts=train_set.groupby('item_id').size().reset_index(name='item_click_counts')

train_set=pd.merge(train_set,user_click_counts,on='user_id',how='left')
train_set=pd.merge(train_set,item_click_counts,on='item_id',how='left')
val_set=pd.merge(val_set,user_click_counts,on='user_id',how='left')
val_set=pd.merge(val_set,item_click_counts,on='item_id',how='left')
train_set['user_click_counts']=train_set['user_click_counts'].fillna(0)
train_set['item_click_counts']=train_set['item_click_counts'].fillna(0)
val_set['user_click_counts']=val_set['user_click_counts'].fillna(0)
val_set['item_click_counts']=val_set['item_click_counts'].fillna(0)

#连续特征离散化（分桶）
bucket_feature=['user_click_counts','item_click_counts','price']
kbd=KBinsDiscretizer(n_bins=10,encode='ordinal',strategy='quantile')
train_set[bucket_feature]=kbd.fit_transform(train_set[bucket_feature])
val_set[bucket_feature]=kbd.transform(val_set[bucket_feature])

#负采样
pos_df=train_set[train_set['label']==1]
neg_df=train_set[train_set['label']==0]

len_pos_df=len(pos_df)
sample_len=min(len_pos_df*10,len(neg_df))

neg_sample=neg_df.sample(sample_len,random_state=42)
train_set=pd.concat([neg_sample,pos_df],axis=0)
train_set=train_set.sample(frac=1,random_state=42)

sparse_features=['item_id','user_id','category_id','user_click_counts','item_click_counts','price']
combined_data=pd.concat([train_set,val_set],axis=0)
for feat in sparse_features:
    lbe=LabelEncoder()
    combined_data[feat]=lbe.fit_transform(combined_data[feat].astype(str))

len_train=len(train_set)
train_set=combined_data.iloc[:len_train].copy()
val_set=combined_data.iloc[len_train:].copy()

fixlen_feature=[
    SparseFeat(feat,vocabulary_size=combined_data[feat].nunique(),embedding_dim=4)
    for feat in sparse_features
]
feature_names=get_feature_names(fixlen_feature)

train_model_input={name:train_set[name].values for name in feature_names}
val_model_input={name:val_set[name].values for name in feature_names}
y_train=train_set['label'].values
y_val=val_set['label'].values
device='cuda' if torch.cuda.is_available() else 'cpu'
model=DeepFM(
    linear_feature_columns=fixlen_feature,
    dnn_feature_columns=fixlen_feature,
    task='binary',
# 正则化 (权重衰减)惩罚过大的权重，强迫模型用更平滑的参数去拟合，而不是走极端
    l2_reg_linear=1e-4,
    l2_reg_embedding=1e-4,
    l2_reg_dnn=1e-4,
# Dropout (随机失活)每次训练随机“戳瞎”一半(0.5)的神经元，强迫剩下的神经元不能偷懒，学到真正鲁棒的特征
    dnn_dropout=0.5,
    device=device
)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['auc'])
model.fit(train_model_input,
          y_train,
          batch_size=512,
          epochs=10,
          verbose=2,
          # 直接喂入验证集字典，代替 validation_split=0
          validation_data=(val_model_input, y_val))
y_pred_prob=model.predict(val_model_input)
auc_score=roc_auc_score(y_val,y_pred_prob)
print(f"AUC Score: {auc_score}")
