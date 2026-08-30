import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
# 注意：这里我们多引入了一个 DenseFeat，专门用来处理连续的数值（如价格）
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names
from deepctr_torch.models import DeepFM
from sklearn.metrics import roc_auc_score

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

dense_feature=['price']
sparse_feature=['category_id','user_id','item_id']
combined_data=pd.concat([train_set,val_set],axis=0)
for feat in sparse_feature:
    lbe=LabelEncoder()
    combined_data[feat]=lbe.fit_transform(combined_data[feat].astype(str))
len_train=len(train_set)
train_set=combined_data[:len_train].copy()
val_set=combined_data[len_train:].copy()
sparse_columns=[
    SparseFeat(feat,vocabulary_size=combined_data[feat].nunique(),embedding_dim=4)
    for feat in sparse_feature
]
dense_columns=[
    DenseFeat(feat,1)
    for feat in dense_feature
]
fixlen_columns=sparse_columns+dense_columns
feature_name=get_feature_names(fixlen_columns)
train_model_input={name:train_set[name].values for name in feature_name}
val_model_input={name:val_set[name].values for name in feature_name}
y_train=train_set['label'].values
y_val=val_set['label'].values
device='cuda'
model=DeepFM(
    linear_feature_columns=fixlen_columns,
    dnn_feature_columns=dense_columns,
    task='binary',
    device=device,
)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['auc'])
history=model.fit(
    train_model_input,
    y_train,
    batch_size=1024,
    epochs=4,
    verbose=2,
    validation_split=0
)
y_pred_prob=model.predict(val_model_input)
auc_score=roc_auc_score(y_val,y_pred_prob)
print(f"AUC Score: {auc_score}")