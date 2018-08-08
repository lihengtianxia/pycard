# -*- coding: utf-8 -*-
"""This module provide some useful tools for universal mission.
Created on Thu May 28 15:06:51 2015 @author: 左词"""
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
from . import config
from time import clock
from datetime import datetime

mdir = lambda obj: [i for i in dir(obj) if not i.startswith('_')]
mdir.__doc__ = "same as dir, but don't collect attributes startswith '_'"

mdiff = lambda col1, col2: [i for i in col1 if i not in col2]
mdiff.__doc__ = "same as set(col1) - set(col2), but keep the order of items in col1"


def re_search(patten, str_set):
    """用正则表达式，在由字符串组成的集合中搜索，返回所有匹配的搜索结果\n
    参数:
    ----------
    patten: str, 正则表达式描述的模式 \n
    str_set: iterable of str, 搜索的目标集合 \n
    返回值:
    ----------
    match: list, 包含所有匹配模式的item
    """
    patten = re.compile(patten)
    match = []
    for i in str_set:
        tmp = patten.search(i)
        if tmp != None:
            match.append(tmp.string)
    return match


def reverse_dict(d):
    """把字典 d 中的 key:value 对反转为 value:key , 返回另一字典. value 必须是不可变的，否则会报错  \n"""
    return {value: key for key, value in d.items()}


def flatten(x):
    """展平嵌套列表结构"""
    out = []
    for i in x:
        if isinstance(i, (list, tuple)):
            out.extend(flatten(i))
        else:
            out.append(i)
    return out


def subset(super_set):
    """返回 super_set 的所有子集"""
    super_set = set(super_set)
    assert len(super_set) >= 2, 'supSet has only one element, so it has no subset'
    sub1 = set()
    subList = set()
    for i in super_set:
        sub1.add(frozenset(super_set - {i}))
        subList.update(sub1)
    if len(super_set) == 2:
        return [set(i) for i in subList]
    else:
        for i in sub1:
            tmp = subset(i)
            tmp = [frozenset(i) for i in tmp]
            subList.update(tmp)
        return [set(i) for i in subList]


def subset_i(super_set, i):
    """返回 supSet 的所有有i个元素的子集"""
    super_set = set(super_set)
    n = len(super_set)
    i_level = n - i
    assert n >= 2, 'supSet has only one element, thus it has no subset'
    assert n >= i, '子集的元素个数不可能超过父集'
    subi = set()
    tmp = set()
    for item in super_set:
        tmp.add(frozenset(super_set - {item}))
    if i_level == 1:
        subi.update(tmp)
        return [set(sub) for sub in subi]
    else:
        for item in tmp:
            a = subset_i(item, i)
            a = [frozenset(sub) for sub in a]
            subi.update(a)
        return [set(sub) for sub in subi]


def pkl_dump(obj, file_path):
    """写 pickle 对象到文件中去"""
    from pickle import dump
    with open(file_path, 'wb') as f:
        dump(obj, f)


def pkl_load(file_path):
    """从 pickle 文件 load 对象"""
    from pickle import load
    with open(file_path, 'rb') as f:
        return load(f)


class Timer:
    """计时装饰器，用来给任意函数计时，以评估其运算效率"""
    def __init__(self, func):
        self.func = func
        self.Alltime = 0
        self.Calls = 0

    def __call__(self, *args, **kwargs):
        start = clock()
        self.Calls += 1
        result = self.func(*args, **kwargs)
        self.Alltime += clock() - start
        self.Meantime = self.Alltime / self.Calls
        return result


def detail(obj):
    """返回一个对象的属性详情。其中\n
    键 'm' 中保存可以调用的属性，一般是方法、类\n
    键 'a' 中保存的是不可调用的属性，一般是普通属性\n
    键 'M' 中保存的是对象中的子模块\n
    键 'u' 中保存的是未知的，getattr出错的属性"""
    tmp = mdir(obj)
    attr = {'m': [], 'a': [], 'u': [], 'M': []}
    for i in tmp:
        try:
            if hasattr(getattr(obj, i), '__call__'):
                attr['m'].append(i)
            elif type(getattr(obj, i)).__name__ == 'module':
                attr['M'].append(i)
            else:
                attr['a'].append(i)
        except:
            attr['u'].append(i)
    attr1 = attr.copy()
    for key in attr1:
        if len(attr1[key]) == 0: attr.pop(key)
    return attr


def class_tree(cls, indent=3):
    """打印出类的继承树出来，缩进量越多的类，是层级越高（越早）的超类"""
    print('.' * indent + cls.__name__)
    for supercls in cls.__bases__:
        class_tree(supercls, indent + 3)


def instance_tree(inst):
    """打印出对象的继承树出来，缩进量越多的类，是层级越高（越早）的超类"""
    print('Tree of %s' % inst)
    class_tree(inst.__class__, 3)


# %% 数据库相关
class MysqlTools(object):
    """创建连接不同数据库的引擎, 方便读写数据库。"""

    def __init__(self, name, conf=config.CONNECT_DICT):
        """参数name : int or str , 数据库连接名，需要提前在config.CONNECT_DICT中配置\n""" + conf['__doc__']
        self.__connect_dict = conf
        assert name in self.__connect_dict, 'Unknown host'
        self.Name = name

    def __con(self):
        """定义专门的私有方法，用于创建引擎"""
        url = self.__connect_dict[self.Name]
        return create_engine(url)

    def query(self, sql):
        """执行查询操作，返回查询的结果表df \n
        参数:
        ----------
        sql: str, 由字符串表达的查询语句"""
        con = self.__con()
        return pd.read_sql_query(sql, con)

    def to_sql(self, df, table_name):
        """把df表中的数据存入数据库中、以table_name为名的表中。若此表已在库中，则会把数据追加在尾部。"""
        con = self.__con()
        df.to_sql(table_name, con, if_exists='append', index=False)

    def exe(self, sql):
        """清空表、删除表、创建表等任意sql操作 \n
        参数:
        ----------
        sql: str, 由字符串表达的数据库语句"""
        con = self.__con()
        return con.execute(sql)

    @property
    def show_tables(self):
        """返回数据库中的所有表"""
        b = self.query('show tables')
        return b

    def desc_table(self, table_name):
        """返回一个表的元数据信息"""
        b = self.query('describe {}'.format(table_name))
        b.Field = b.Field.str.lower()
        return b


class DBinfo:
    __doc__ = """把数据库的元信息读出用于搜索了解。主要信息有：数据库的表、视图，每个表的字段详情。\n
    db_name：int, 数据库编码，在config中配置. 已配置值：\n""" + config.CONNECT_DICT['__doc__']

    def __init__(self, db_name):
        url = config.CONNECT_DICT[db_name]
        con = create_engine(url)
        self.Tables = pd.read_sql_query("show tables", con)
        self.Details = []
        self.Err_table = []

        col0 = self.Tables.columns[0]
        priKey = {}
        for table in self.Tables[col0]:
            try:
                tb_detail = pd.read_sql_query("DESCRIBE " + table, con)
                tb_detail['tbName'] = table
                self.Details.append(tb_detail)

                key = tb_detail.loc[tb_detail.Key == 'PRI', 'Field']
                if len(key) == 1:
                    key = key[0]
                elif len(key) > 1:
                    key = ','.join(list(key))
                else:
                    key = np.nan
                priKey[table] = key
            except:
                self.Err_table.append(table)
        key = pd.DataFrame({'priKey': priKey})
        self.Tables = self.Tables.merge(key, left_on=col0, right_index=True)
        self.Details = pd.concat(self.Details, ignore_index=True)

    def find_tb(self, tablename):
        """找出表名符合正则表达式的所有表\n
        tablename: str, 可以是普通字符串，也可以是正则表达式描述的表名"""
        col0 = self.Tables.columns[0]
        tbList = re_search(tablename, self.Tables[col0])
        return tbList

    def find_col(self, colname):
        """找出符合正则表达式的所有列名\n
        colname: str,可以是普通字符串，也可以是正则表达式描述的列名"""
        colList = re_search(colname, self.Details['Field'])
        if colList:
            logic = self.Details.Field.isin(colList)
            tmp = self.Details[logic]
        return tmp

    def desc_table(self, tablename):
        """返回指定名称的表的详细信息"""
        logic = self.Details.tbName == tablename
        return self.Details[logic]

    def __repr__(self):
        col = self.Tables.columns[0].split('_')
        db = col[-1]
        tbN = len(self.Tables)
        colN = len(self.Details)
        return 'database: {0}, table_num: {1}, col_num: {2}'.format(db, tbN, colN)


def infer_mysql_dtype(df, frac=0.3, print_flag=True):
    """根据列的抽样数据，推断各列的mysql数据类型, 以便生成建表代码 \n
    参数:
    ----------
    df:  需要推断的数据框 \n
    frac: float, 抽样比例。推断将在按此比例抽样的子表数据上进行 \n
    print_flag: bool, 是否打印出打断的结果字符串，如果为Ture, 就打印字符串并返回None；如果为False，则不打印但返回字符串"""
    known_col = config.MYSQL_COL_DTYPE

    def real_int(sr):
        """推测有空值的float64类型的列，是否实际上是int类型。"""
        logic = sr.isnull()
        if logic.all():
            return False
        else:
            sr = sr[~logic]
            diff = (sr - sr.map(int)).abs()
            return (diff == 0).all()

    column_types = pd.Series();
    dtypes = df.dtypes
    df_infer = df.sample(frac=frac)
    for col in dtypes.index:
        if col in known_col:
            column_types[col] = known_col[col]
        else:
            dtype = dtypes[col]
            if np.issubdtype(dtype, np.integer):
                column_types[col] = 'int'
            elif np.issubdtype(dtype, np.floating):
                column_types[col] = 'int' if real_int(df_infer[col]) else 'float'
            elif np.issubdtype(dtype, np.datetime64):
                column_types[col] = 'datetime'
            elif dtype == np.dtype('bool'):
                column_types[col] = 'bool'
            else:  # dtype('O')
                if col.lower().find('date') > -1:
                    column_types[col] = 'date'
                elif col.lower().find('time') > -1:
                    column_types[col] = 'datetime'
                else:
                    char_len = df[col].map(lambda x: len(x) if pd.notnull(x) else 0).max()
                    column_types[col] = 'varchar({})'.format(char_len)  # 最大长度是基于所有样本数据计算得到的
    a = '\n'.join("{0} {1},".format(key, column_types[key]) for key in df)
    if print_flag:
        print(a)
    else:
        return a


def select_except(table, col, con, alias=None):
    """生成除了 col 列以外的所有列的 select 代码。当一个表有很多列，而除了少数几列外，其他所有列都需要查询时，用此函数 \n
    生成查询的列比手动敲各个列要方便很多。 \n
    table: 需要select的表名\n
    col: 不希望select的字段名\n
    con: MysqlTools 对象"""
    sqlStr = "describe " + table
    tableInfo = con.query(sqlStr)
    field = tableInfo.Field.str.lower()
    if isinstance(col, str):
        col = col.lower()
        logic = field != col  # 剔除单个列
    else:
        col = [i.lower() for i in col]
        logic = ~field.isin(col)  # 剔除多个列组成的序列
    tableFilter = tableInfo[logic]
    if alias is None:  # 表的别名
        alias = table
    field = tableFilter.Field.apply(lambda x: alias + "." + x)
    field = ",\n".join(field)
    print(tableInfo)
    return field


def query_period(stime, etime):
    """返回一个函数，此函数统一给sql查询加上起止日期\n
    stime: str or date, 起始日期\n
    etime: str or date, 终止日期"""

    def func(sql_str, date_col, date_length=10):
        """给sql查询语句增加上起止日期 \n
        sql_str: str, 查询语句 \n
        date_col: 日期或日期时间的字段名，起止日期即由此字段的值决定。 \n
        date_length: int, 日期的格式。取值8表示'20170501'格式的日期，取值10表示'2017-05-01'格式的日期. """
        if date_length == 8:
            apd = " and {2} BETWEEN '{0}' and '{1}'""".format(
                ''.join(stime.split('-')), ''.join(etime.split('-')), date_col)
        else:
            apd = " and {2} BETWEEN '{0}' and '{1}'""".format(stime, etime, date_col)
        sql_str_lower = sql_str.lower()
        try:
            sql_str_lower.index('where')
        except ValueError:  # 没有 where 子句
            apd = " where" + apd[4:]
        try:
            loc = sql_str_lower.index('group by')
        except ValueError:
            try:
                loc = sql_str_lower.index('order by ')
            except ValueError:
                try:
                    loc = sql_str_lower.index('limit ')
                except ValueError:
                    return sql_str + apd
        sql_apd = sql_str[:loc] + apd + sql_str[loc:]
        return sql_apd

    return func


def melt_na(frame, id_vars=None, value_vars=None, var_name=None, value_name='value',
            col_level=None):
    """把宽表变成长表，然后删除na行，返回紧凑的长表"""
    longDf = pd.melt(frame, id_vars=id_vars, var_name=var_name, value_vars=value_vars,
                     value_name=value_name, col_level=col_level)
    return longDf[longDf[value_name].notnull()]


def foreach(func, iterator):
    """依次对iterator中的所有元素应用func函数。没有返回值\n
    参数:
    ----------
    func: 任意函数，以iterator中的元素为输入参数，无返回值 \n
    iterator: 任意可迭代对象"""
    for item in iterator:
        func(item)


def prteach(iterator):
    """依次打印集合元素中的元素 \n
    参数:
    ----------
    iterator: 任意可迭代对象"""
    foreach(print, iterator)


def loads_json(json_str):
    """加载 json 字符串为 python 对象。出错就根据格式推断，返回空的 list 或 dict \b
    参数:
    ----------
    json_str: json 格式的字符串"""
    from json import loads
    try:
        x = loads(json_str)
    except:  # 格式错误的话，返回空的 list 或 dict
        x = [] if json_str[:30].strip().startswith('[') else {}
    return x


def dical_prod(set1, set2, return_type='string'):
    """把 set1 和 set2 中的元素做笛卡尔积组合。如果set1有m个元素，set2有n个元素，则组合结果有m*n个元素 \n
    参数:
    ----------
    set1, set2 : 任意可迭代的对象，需要组合的两个对象。  \n
    return_type: str, 'string' 表示把2个元素合并成一个字符串，以'list' 表示把2个元素放在一个list中  \n """
    dical = [[i, j] for i in set1 for j in set2]
    if return_type == 'string':
        dical = [item[0] + '_' + item[1] for item in dical]
    return dical


def doc_generate(doc,params,returns):
    """生成函数、类的说明文档，此文档是标准化格式的。\n
    参数:
    ----------
    doc: str, 函数、类的功能性说明文档 \n
    params: dict of {param_name: param_doc}, param_name是函数、类的输入参数名字，param_doc是关于此参数的类型、含义的说明文档 \n
    returns: dict of {return_name: return_doc}, return_name是返回值的名字，return_doc是关于此返回值的类型、含义的说明文档 \n
    返回值:
    ----------
    无返回值，会把生成的文档打印到屏幕"""
    fun_format = """
    {doc}
    
    输入参数:
    -----------
    {params}
    
    返回值:
    -----------
    {returns}
    """
    
    params_str = "{name}: {doc}"
    params_str_list = []
    for name,doc_p in params.items():
        params_str_list.append(params_str.format(name=name, doc=doc_p))
    
    returns_str_list = []
    for name,doc_t in returns.items():
        returns_str_list.append(params_str.format(name=name, doc=doc_t))
    
    fun_doc = fun_format.format(doc=doc, 
                                params='\n'.join(params_str_list),
                                returns='\n'.join(returns_str_list))
    print(fun_doc)
    
    
def strptime(str_date):
    """把字符串形式的datetime, 转换成 datetime 类型。\n
    输入参数:
    -----------
    str_date: str, 2018-03-12 12:28:32 格式的时间
    返回值:
    -----------
    dtime: datetime, 转换后的 datetime 对象"""
    return datetime.strptime(str_date,'%Y-%m-%d %H:%M:%S')


