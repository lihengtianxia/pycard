# -*- coding: utf-8 -*-
"""Created on Mon Mar  5 20:58:25 2018 @author: 左词
模块的配置文件. 修改这里的配置文件后，需要重新加载整个模块以使修改生效"""

# 拼表中，最常用的列名，比如身份证号码。pycard将以此列名定制2个拼表方法，新增给DataFrame:
# pd.DataFrame.merge_{colname}_left : 以 colname 为拼接键，左连接
# pd.DataFrame.merge_{colname}_inner ：以 colname 为拼接键，内连接
MERGE_KEYS = ['contract_id', 'idcard', 'user_id']

# 最主要的主键，评分卡一般在合同级别或人的级别上建模，因此最主要的主键一般是合同ID或人的ID。
# 最主要的主键一般是在拼表等各处中用到最多的，一些函数经常用到这一列。
PRIM_KEY = 'contract_id'

# 数据库账号 URL，MysqlTools使用此配置信息以读写指定的库
CONNECT_DICT = {'__doc__': """
0 -- test库;        'rmps'  --  风控库。""",
0:   'mysql+pymysql://root:shj1234@127.0.0.1:3306/test?charset=utf8',
'rmps':   'mysql+pymysql://root:shj1234@127.0.0.1:3306/rmps?charset=utf8'}

# 数据库表中常见的 {字段:数据类型} ，配置此字典后，infer_mysql_dtype函数对df的对应列的
# 数据类型便是准确的，不必再推测。
MYSQL_COL_DTYPE = {
'contract_id': 'char(16)', 'idcard_no':'char(18)',
'user_id': 'char(13)','bank_card_no':'char(19)',
'mobile': 'char(11)','aplDate':'date'}

# 做模型需要先把变量分为（类别型,数字型,主键,目标变量,日期时间）几类，此字典配置常见字段
# 所属的类，infer_type函数会利用此信息
MODEL_TYPE = {
'cate':{},  # 类别型
'num':{},  # 数值型
'id':{'contract_id', 'idcard_no', 'user_id', 'bank_card_no', 'bankcard_no',  # 主键类
       'customer_id', 'certid', 'seq_id', 'card_id', 'mobile', 'phone', 'cell'},
'y': {},  # 目标变量
'datetime':{}  # 日期时间
}

