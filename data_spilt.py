import  pandas as pd
from sqlalchemy import create_engine
# ==========================================
DB_USER = 'root'
DB_PASSWORD = '001001'  # 你的 MySQL 密码
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'ecommerce_db'  # 你的数据库名

connection_string = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(connection_string)
sql_query ="""
select *
from user_behavior
order by behavior_time asc
limit 100000
"""
data = pd.read_sql(sql_query, engine)
tota_row=len(data)
train_end=int(tota_row*0.8)
val_end=int(tota_row*0.9)
train_set=data.iloc[:train_end]
val_set=data.iloc[train_end:val_end]
test_set=data.iloc[val_end:]
print(f"✅ 训练集 (Train) 规模: {len(train_set)} 行")
print(f"✅ 验证集 (Val) 规模:   {len(val_set)} 行")
print(f"✅ 测试集 (Test) 规模:  {len(test_set)} 行")
train_set.to_csv("train_set.csv", index=False)
val_set.to_csv("val_set.csv", index=False)
test_set.to_csv("test_set.csv", index=False)