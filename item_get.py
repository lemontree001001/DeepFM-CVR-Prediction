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
sql_query = """
select a.item_id,b.category_id,b.price 
from (select item_id from user_behavior order by behavior_time limit 100000) a
left join item_info b on a.item_id=b.item_id
"""
item_set= pd.read_sql(sql_query, engine)
item_set.to_csv("item_set.csv", index=False)