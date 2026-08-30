import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

train_set=pd.read_csv('./train_set.csv')
val_set=pd.read_csv('./val_set.csv')

train_set['label']=(train_set['behavior_type']=='alipay').astype(int)
val_set['label']=(val_set['behavior_type']=='alipay').astype(int)

pos_weight = (len(train_set) - train_set['label'].sum()) / train_set['label'].sum()
print(f"正负样本失衡权重 (scale_pos_weight) 设定为: {pos_weight:.2f}")

print("3. 提取特征与 Label Encoding...")
sparse_feature=['user_id','item_id','behavior_type']
combined_data=pd.concat([train_set,val_set],axis=0)
for feat in sparse_feature:
    lbe=LabelEncoder()
    combined_data[feat]=lbe.fit_transform(combined_data[feat].astype(str))

len_train=len(train_set)
train_set=combined_data.iloc[:len_train].copy()
val_set=combined_data.iloc[len_train:].copy()
drop_columns=['behavior_time','label','behavior_type']
x_train=train_set.drop(columns=drop_columns)
y_train=train_set['label']
x_val=val_set.drop(columns=drop_columns)
y_val=val_set['label']
print("4. 唤醒 GPU 算力，启动 XGBoost 训练...")
xgb_model = XGBClassifier(
    n_estimators=150,           # 树的数量
    learning_rate=0.1,          # 学习率
    scale_pos_weight=pos_weight,# 处理极其不平衡的电商数据
    tree_method='hist',         # 高效的直方图算法
    device='cuda',              # 直接拉起 RTX 算力进行加速
    random_state=42
)
xgb_model=xgb_model.fit(x_train,y_train)
# predict_proba 会返回 0 和 1 的概率，[:, 1] 表示提取预测为 1（购买）的概率
y_pred_prob=xgb_model.predict_proba(x_val)[:,1]

auc_score=roc_auc_score(y_val,y_pred_prob)
print(f"🎯 最终验证集 AUC 跑分: {auc_score:.4f}")