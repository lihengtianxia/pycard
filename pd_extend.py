# -*- coding: utf-8 -*-
"""Created on Thu Nov 23 20:53:19 2017 @author: 左词
扩展了 pandas 模块，添加了开发评分卡经常用到的方法"""
import numpy as np
import pandas as pd
import shelve as sv
from .tools import re_search
from . import config


def default_param_decorator(func, **my_kwargs):
    """默认参数装饰器，带有默认参数值的函数和方法，修改其默认参数的值  \n
    参数:
    ---------
    func: 函数，其中带有若干个默认参数，但默认参数设定的值不是最常用的值，期望装饰器更改它  \n
    my_kwargs: 若干个关键字参数，其默认值可以设定成所期望的值   \n
    返回值:
    ---------
    func_param: 函数，与 func 功能相同，只是由my_kwargs设定的关键字参数的默认值不同"""
    def func_param(*args, **kwargs):
        kwargs.update(my_kwargs)
        return func(*args, **kwargs)
    return func_param


pd.DataFrame.drop_col = default_param_decorator(pd.DataFrame.drop, axis=1)
pd.DataFrame.drop_col.__doc__ = """
same as df.drop(labels,axis=1,**kwargs), 仅针对表的column定义的删除方法。\n
Parameters:
-----------
labels : single label or list-like, 需要删除的列名。\n
其余所有参数，参见df.drop方法"""

pd.DataFrame.to_csv_idx = default_param_decorator(pd.DataFrame.to_csv, index=False)
pd.DataFrame.to_csv_idx.__doc__ = """
same as df.to_csv(path_or_buf,index=False,**kwargs)，不保存df的index。\n
Parameters:
-----------
path_or_buf : string or file handle, default None, will return the result as a string.\n
其余所有参数，参见df.to_csv方法"""

pd.DataFrame.to_sql_idx = default_param_decorator(pd.DataFrame.to_sql, index=False,
                                                  if_exists='append', flavor='mysql')
pd.DataFrame.to_sql_idx.__doc__ = """
same as df.to_sql(name, con, flavor='mysql', index=False, if_exists='append',**kwargs)\n
把 df 中的数据写入到数据库的指定表中。\n
Parameters:
-----------
name : str, 数据库表名。若此表已存在，会把数据追加在表的尾部。\n
con : SQLAlchemy engine，数据库连接对象。\n
其余所有参数，参见df.to_sql方法"""


def add_merge_keys():
    """为 DataFrame 添加 merge 相关的方法"""
    for colname in config.MERGE_KEYS:
        for how in ['left', 'inner']:
            fun_name = 'merge_{on}_{how}'.format(on=colname, how=how)
            setattr(pd.DataFrame,
                    fun_name,
                    default_param_decorator(pd.DataFrame.merge, on=colname, how=how))
            setattr(getattr(pd.DataFrame, fun_name),
                    '__doc__',
                    "same as df.merge(other,on='{on}',how='{how}')".format(on=colname, how=how))
add_merge_keys()


def default_keyparam_decorator(func, *ex_args):
    """默认关键字参数装饰器，此装饰器把传入的参数自动赋值给args指定的关键字参数名。\n
    参数:
    ----------
    func: 函数，其中有些关键字参数，经常需要给它们传值。  \n
    ex_args: str, 关键字参数的名字，此装饰器自动把传入的值，传递给func的这些关键字参数  \n
    返回值:
    ----------
    func_key: 函数，逻辑同 func, 只是把 func 中由ex_args指定的那些关键字参数当作位置参数来使用。  \n
    Example:
    --------
    pd.DataFrame.rename_col = defaultKeyParam(pd.DataFrame.rename,'columns')
    d = {'oldname1':'newname1','oldname2':'newname2'}
    df.rename_col(d)  # 等价于 df.rename(columns=d), 懒得在每次调用 rename 方法时敲 columns="""
    def func_key(*args, **kwargs):
        n = len(ex_args)
        args_for_key = args[-n:]  # 传给默认关键字的参数，要按顺序放在位置参数的后面，这样args的最后n个参数，
        default_key = dict(zip(ex_args, args_for_key))  # 就能与ex_args_key里面的默认关键字参数对应起来
        kwargs.update(default_key)  
        if len(args) > n:  # 如果除了默认关键字参数，还有其他位置参数
            args = args[:-n]  # 提取 args 里面的位置参数
            return func(*args, **kwargs)
        else:
            return func(**kwargs)
    return func_key
pd.DataFrame.rename_col = default_keyparam_decorator(pd.DataFrame.rename, 'columns')
pd.DataFrame.rename_col.__doc__ = """
重命名columns, df.rename_col(d) 等价于 df.rename(columns=d)\n
Parameters:
-----------
d : dict of {oldname:newname}, 需要重命名列名的字典。\n"""


def apply_horizon(self, func, **kwargs):
    """水平方向（行方向）apply，并且返回结果是df \n
    参数:
    ----------  
    func: 以df的一行为输入，以一个字典为返回值的函数. func若返回Series，计算会慢很多\n
    kwargs: 其他传给 apply 方法的参数\n
    返回值:
    ----------
    applyed_df: DataFrame, columns由func返回字典的键组成，每一行的值由func返回字典的值组成
    """
    srOfDictElements = self.apply(func, axis=1, **kwargs)
    return pd.DataFrame(srOfDictElements.tolist())
pd.DataFrame.apply_horizon = apply_horizon


def idrop_axis(self, ints):
    """
    以整数值来指定需要删除的行。返回删除指定行之后的df。\n
    drop 方法以 labels 来指定需要删除的行，无法以labels的序号来指定需要删除的行.\n
    参数:
    ----------  
    ints: int or list_of_int, 需要删除的行的序号\n
    see also: idrop, idrop_col, drop, drop_col
    """
    idx = list(range(len(self)))
    if isinstance(ints, int): ints = [ints]
    for i in ints:
        idx.remove(i)
    return self.iloc[idx, :]
pd.DataFrame.idrop_axis = idrop_axis


def idrop_col(self, ints):
    """
    以整数值来指定需要删除的列。返回删除指定列之后的df。\n
    参数:
    ----------    
    ints: int or list_of_int, 需要删除的列的序号 \n
    see also: idrop, idrop_axis, drop, drop_col \n
    """
    idx = list(range(len(self.columns)))
    if isinstance(ints, int): ints = [ints]
    for i in ints:
        idx.remove(i)
    return self.iloc[:, idx]
pd.DataFrame.idrop_col = idrop_col


def idrop(self, ints, axis=0):
    """调用 idro_axis 或 idrop_col，返回删除指定列或行之后的df \n
    参数:
    ----------    
    ints: int or list_of_int, 需要删除的行/列的序号 \n"""
    if axis == 0:
        return self.idrop_axis(ints)
    elif axis == 1:
        return self.idrop_col(ints)
pd.DataFrame.idrop = idrop


def corr_tri(self):
    """返回相关性矩阵的下三角矩阵, 上三角部分的元素赋值为nan \n"""
    corr_df = self.corr()
    n = len(corr_df)
    logic = np.ones((n,n))
    for i in range(n):
        for j in range(n):
            if i > j:
                logic[i,j] = 0
    return corr_df.mask(logic == 1)
pd.DataFrame.corr_tri = corr_tri


def nan_rate(self):
    """计算缺失率, 返回值是个序列"""
    d = pd.Series()
    d['count'] = self.count()
    d['len'] = len(self)
    d['nanRate'] = 1 - d['count'] / d['len']
    return d
pd.Series.nan_rate = nan_rate


def nan_rate_df(self, col=None):
    """计算各列的缺失率\n
    参数:
    ----------    
    col: 可以是单个列字符串，或列的集合. 需要计算缺失率的列，None表示计算所有列 \n
    返回值:
    ----------    
    当col只有一列时，返回序列；当col是多列时，返回df"""
    if col==None: col=self.columns
    if len(col) == 1 or isinstance(col,str): # col只有一列的话
        return self[col].nan_rate()
    else:
        d = []
        for i in col:
            nan_sr = self[i].nan_rate()
            nan_sr.name = i
            d.append(nan_sr)
        return pd.concat(d, axis=1).transpose()
pd.DataFrame.nan_rate = nan_rate_df


def describe_3std(self):
    """在describe计算的统计量基础上，还计算3倍标准差的左右边界值，看哪些观测超出了边界值。\n
    返回表的error列中，取值-1表示最小值在左边界外，取值1表示最大值在右边界外"""
    des = self.describe().transpose()
    des['left_edge'] = des['mean'] - 3 * des['std']
    des['right_edge'] = des['mean'] + 3 * des['std']
    des['error'] = 0
    des.loc[des['min'] < des['l_edge'], 'error'] = -1
    des.loc[des['max'] > des['r_edge'], 'error'] = 1
    return des
pd.DataFrame.describe_3std = describe_3std


def dup_info(self):
    """对元素值进行重复性检查，并统计重复值信息"""
    class DupInfo:
        def __init__(self, name):
            self.name = name
            self.count, self.sum = 0, 0

        def __str__(self):
            return 'name: %s\ncount: %s\nsum: %s\n' % (self.name, self.count, self.sum)

        def __repr__(self):
            return self.__str__()

    dup = DupInfo(self.name)
    logic = self.duplicated()
    if logic.sum():
        dup.rows = self[logic].unique() 
        dup.logic = self.isin(dup.rows)
        dup.count = len(dup.rows)
        dup.value_counts = self[dup.logic].value_counts()
        dup.sum = dup.value_counts.sum()
    return dup
pd.Series.dup_info = dup_info
pd.Index.dup_info = dup_info


def dup_col(self, drop= None):
    """如果两个表中有同名的列，则拼表后，同名的列会被自动重命名为colname_x,colname_y \n
    此方法识别带'_x'和'_y'后续的列名，并重命名回以前的名字。\n
    参数:
    ----------
    drop: 值为'x'则删除colname_x列; 值为'y'则删除colname_y列；\n
        默认值None表示打印出重复列的相关信息，不删除列。\n
    返回值:
    ----------
    重命名后的df，不修改原数据框
    """
    x = re_search('_x$', self)
    y = re_search('_y$', self)
    if drop == 'y':
        col = {i: i[:-2] for i in x}
        drop_col = y
    elif drop == 'x':
        col = {i: i[:-2] for i in y}
        drop_col = x
    else:        
        print(self[x].count())
        print(self[y].count())
        return
    return self.drop(drop_col, axis=1).rename(columns=col)
pd.DataFrame.dup_col = dup_col


def same_col(self, drop=None):
    """表中有完全重名的列时，删除指定的列。返回新的df，不修改原df \n
    参数:
    ----------
    drop: int or list_of_ints, 需要删除的重名列的序号（完全重名的列无法以label定位）\n
        默认值None表示打印出重名列的相关信息，不删除列。"""
    dup = self.columns.dup_info()
    if dup.count == 0:
        print("没有重复的列名")
        return
    a = np.arange(len(self.columns))
    ind = a[dup.logic]
    info = self.iloc[:, ind].count().reset_index()
    info['ind'] = ind
    if drop is None:
        return info
    else:
        return self.idrop_col(drop)
pd.DataFrame.same_col = same_col


def distribution(self, sort='index'):
    """计算单个变量的分布, 返回的数据框有两列：cnt(个数)、binPct(占比) \n
    参数:
    ----------
    sort: str, 'index' 表示按变量值的顺序排序，其他任意值表示变量值的个数占比排序"""
    a = self.value_counts()
    b = a / a.sum()
    df = pd.DataFrame({'cnt': a, 'binPct': b})
    if sort == 'index':
        df = df.sort_index()
    return df.reindex(columns=['cnt', 'binPct'])
pd.Series.distribution = distribution


def destribution_df(self, col=None):
    """计算df的各个变量的频率分布。\n
    参数:
    ----------
    col: str or list_of_str, 需要计算频率分布的列，None表示所有列。col不应包含\n
        连续型的列，只应包含离散型的列。
    """
    if col is None:
        col = self.columns
    if isinstance(col, str): col = [col]
    var_cnts = pd.DataFrame()
    for i in col:
        di = self[i].destribution()
        di['colName'] = i
        var_cnts = var_cnts.append(di)
    return var_cnts
pd.DataFrame.distribution = destribution_df


def cv(self):
    """计算变异系数。"""
    m = self.mean()
    s = self.std()
    if m == 0:
        print("warning, avg of input is 0")
    return s / m
pd.Series.cv = cv


def argmax(self):
    """计算dataframe中最大值的行、列索引值，返回(row,col)元组"""
    col = self.max().argmax()
    row = self.max(axis=1).argmax()
    return row, col
pd.DataFrame.argmax = argmax


def argmin(self):
    """计算dataframe中最小值的行、列索引值，返回(row,col)元组"""
    col = self.min().argmin()
    row = self.min(axis=1).argmin()
    return row, col
pd.DataFrame.argmin = argmin


def pivot_tables(self, values, index, columns):
    """定制的透视表，把badRate、样本数、坏的个数、样本占比一次全求出来"""
    tb = []    
    for fun in ['mean', 'count', 'sum']:
        a = self.pivot_table(values=values, index=index, columns=columns, margins=True, aggfunc=fun)
        tb.append(a)
    tb.append(tb[1]/tb[1].iloc[-1, -1])
    return pd.concat(tb)
pd.DataFrame.pivot_tables = pivot_tables


def ranges(self):
    """计算最小值、最大值"""
    minv, maxv = self.min(), self.max()
    if isinstance(self, pd.DataFrame):
        return pd.DataFrame({'min': minv, 'max': maxv})
    else:
        return pd.Series({'min': minv, 'max': maxv})
pd.DataFrame.range = ranges
pd.Series.range = ranges


def value_counts(self, col=None):
    """df 对多个列的value_counts，返回一个df.\n
    参数:
    ----------
    col: iterable of col_names, 默认值None表示对df的所有列计算"""
    if col is None:
        col = self.columns
    vc = []
    for i in col:
        vci = self[i].value_counts().to_frame(name='count')
        vci['colname'] = i
        vc.append(vci)
    return pd.concat(vc, axis=0)
pd.DataFrame.value_counts = value_counts


def group_distribution(self, col, by):
    """计算分组后的离散变量的分布,col应是单个列名,by可以是组合字段或单个字段\n
    参数:
    ----------
    col : str or list_of_str, 需要计算分布的列\n
    by: 用来分组的列，可以是列名、等长的数组/序列\n
    see also pivot_table"""
    g = self.groupby(by)
    dest = g[col].value_counts().unstack().transpose()
    dest = dest / dest.sum()
    return dest
pd.DataFrame.group_distribution = group_distribution


def group_describe(self, col, by):
    """计算分组后的连续变量的describe，col和by都可以是组合字段或单个字段 \n
    参数:
    ----------
    col : str or list_of_str, 需要计算常规统计的列\n
    by: 用来分组的列，可以是列名、等长的数组/序列"""
    g = self.groupby(by)
    desc = g[col].describe().unstack().transpose()
    return desc
pd.DataFrame.group_describe = group_describe


def unique_dtype(self):
    """计算 Series 非重复的数据类型个数"""
    return self.map(type).value_counts()
pd.Series.unique_dtype = unique_dtype


def isreal(self):
    """判断 dtype 是否是实数类型，bool_, integer, floating 是实数，其他都不是"""
    return self.dtype in [np.integer, np.floating, np.bool_]
pd.Series.isreal = isreal


def find_idx(self, label):
    """找出 label 在 index 中的位置 \n
    参数:
    ----------
    label: 需要在 index 中查找的值 \n
    返回值:
    ----------
    idx: int, 若查中，则返回 label 在 index 中的下标；若未查中，返回-1"""

    def op1(labeli):
        if pd.isnull(label):  # label是缺失的话，用==判断找不出来
            return pd.isnull(labeli)  # is np.nan 判断缺失不可靠，np.isnan函数判断才可靠, 因为 is 是判断同一性
        else:
            return labeli == label

    for i, labeli in enumerate(self):
        if op1(labeli):
            return i
    return -1
pd.Index.find = find_idx


# shelve 增强
def write(self, var_name):
    """把名为 var_name 的变量写入shelve文件中。当要保存多个变量时，使用此方法更方便\n
    参数:
    -----------
    var_name: str or list_of_str, 需要保存数据的变量的名字\n
    示例:
    -----------
    self.write(['a','b']) 等同于下列代码： \n
    self['a'] = a \n
    self['b'] = b """
    if isinstance(var_name, str):   # 如果是单个变量
        var_name = [var_name]
    for i in var_name:
        exec('self["' + i + '"] = ' + i)
sv.DbfilenameShelf.write = write


def dec(fun):
    def keys(self):
        """返回由键组成的list"""
        k = fun(self)  # fun 是旧的sv.DbfilenameShelf.keys函数
        return list(k)  # k 是个KeyView，无法直接查看具体的items
    return keys
sv.DbfilenameShelf.keys = dec(sv.DbfilenameShelf.keys)


#%% 删除多余的函数
del pivot_tables, distribution, same_col, dup_col, dup_info, ranges, corr_tri, value_counts
del apply_horizon, unique_dtype, nan_rate, nan_rate_df, idrop, idrop_axis, idrop_col
del group_distribution, group_describe, default_param_decorator, default_keyparam_decorator
del cv, describe_3std, destribution_df, write, dec, isreal, add_merge_keys
del argmax, argmin, find_idx

