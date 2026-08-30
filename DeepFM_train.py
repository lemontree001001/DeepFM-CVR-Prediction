import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, get_feature_names

train_set=pd.read_csv('./train_set.csv')
val_set=pd.read_csv('./val_set.csv')

train_set['label']=(train_set['behavior_type']=='alipay').astype(int)
val_set['label']=(val_set['behavior_type']=='alipay').astype(int)

sparse_feat=['user_id','item_id']
combined_data=pd.concat([train_set,val_set],axis=0)
for feat in sparse_feat:
    lbe=LabelEncoder()
    combined_data[feat]=lbe.fit_transform(combined_data[feat])
len_train=len(train_set)
train_set=combined_data.iloc[:len_train].copy()
val_set=combined_data.iloc[len_train:].copy()

fixlen_feature_columns=[
        SparseFeat(feat,vocabulary_size=combined_data[feat].nunique(),embedding_dim=4)
    for feat in sparse_feat
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
history=model.fit(
    train_model_input,
    y_train,
    batch_size=512,
    epochs=5,
    verbose=2,
    validation_split=0
)
y_pred_prob=model.predict(val_model_input)
auc_score=roc_auc_score(y_val,y_pred_prob)
print(f"AUC Score: {auc_score}")