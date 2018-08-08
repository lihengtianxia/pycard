# -*- coding: utf-8 -*-
"""Created on Thu May 28 15:06:51 2015 @author: 左词
评分卡包的主模块, 包含了数据清洗、转换相关的所有工具"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.distributions import chi2
from sklearn.tree import DecisionTreeClassifier
from datetime import date
from sklearn.metrics import auc
from . import config, tools
from os.path import join
from sklearn.cross_validation import train_test_split


def infer_type(detail_df):
    """推测出 detail_df 各列的数据类型, 归为 (类别型,数字型,主外键,目标变量,日期时间) 5类. \n
    此函数无法推测序数型变量. \n
    参数:
    ----------
    detail_df: dataframe, 一般是用来开发评分卡的明细表，每个观测一行，每个特征一列 \n
    返回值:
    ----------
    type_df: dataframe, 函数推测的各列的数据类型，包括 2 列：'dtype' 列是pandas类型；'infered' 列是函数推断的类型"""
    ids = config.MODEL_TYPE['id']
    ys = config.MODEL_TYPE['y']
    cate = config.MODEL_TYPE['cate']
    dtime = config.MODEL_TYPE['datetime']
    num = config.MODEL_TYPE['num']

    dtypes = detail_df.dtypes
    col_type = dtypes.copy()
    for col_name, dtype in zip(dtypes.index, dtypes):
        # config中配置的分类
        if col_name in ids:
            col_type[col_name] = 'id'
        elif col_name in ys:
            col_type[col_name] = 'y'
        elif col_name in cate:
            col_type[col_name] = 'cate'
        elif col_name in dtime:
            col_type[col_name] = 'dtime'
        elif col_name in num:
            col_type[col_name] = 'num'

        # 推测的分类
        elif (np.issubdtype(dtype, np.integer) or
                  np.issubdtype(dtype, np.floating)):
            col_type[col_name] = 'num'
        elif (col_name.lower().find('date') > -1 or
                      col_name.lower().find('time') > -1 or
                      col_name.lower().find('week') > -1 or
                  np.issubdtype(dtype, np.datetime64)):
            col_type[col_name] = 'dtime'
        else:
            col_type[col_name] = 'cate'
    type_df = pd.concat([dtypes, col_type], axis=1)
    return type_df.rename(columns={0: 'dtype', 1: 'infered'})


def unique_k(detail_df, k=6):
    """记一列的非重复值个数为 Un, 此函数选出 Un <= k 的列，返回明细。\n
    参数:
    ----------
    detail_df: dataframe, 
    k: int, 非重复值个数的阈值 \n
    返回值:
    ----------
    colk: dataframe, 由低于阈值的列组成的数据框，数据框有3列：列名、Un、列数据类型。"""
    col = []
    Un = []
    col_type = []
    for i in detail_df.columns:
        Un.append(len(detail_df[i].unique()))
        col.append(i)
        col_type.append(detail_df[i].dtype)
    col_unq = pd.DataFrame({'colName': col, 'unique': Un, 'dtype': col_type})
    colk = col_unq[col_unq['unique'] <= k]
    return colk


class Col:
    """用以保存不同数据类型的列，并随着分析不断地记录删除的变量.\n
    基本操作：cate, num, ordi, drop, add_drop, remove_drop, add。\n
    初始化参数:
    -----------
    cate=() : iterable, 类别型变量的集合。\n
    num=() : iterable, 数值型变量的集合。\n
    ordi=() : iterable, 序数型变量的集合。 \n
    drop=() : iterable, 需要剔除的变量的集合  """

    def __init__(self, cate=(), num=(), ordi=(), drop=()):
        self.__cate = list(cate)  # 保存无序分类变量        
        self.__num = list(num)  # 保存数值变量
        self.__ordi = list(ordi)  # 保存序数型变量
        self.__drop = list(drop)  # 保存被删除的变量

    def cate(self):
        """返回尚未被剔除的类别型变量组成的列表"""
        return [i for i in self.__cate if i not in self.drop()]

    def num(self):
        """返回尚未被剔除的数值型变量组成的列表"""
        return [i for i in self.__num if i not in self.drop()]

    def ordi(self):
        """返回尚未被剔除的序数型变量组成的列表"""
        return [i for i in self.__ordi if i not in self.drop()]

    def drop(self):
        """返回已被剔除的所有变量"""
        return self.__drop

    def add_drop(self, name):
        """添加需要剔除的列名. 无返回值 \n
        参数:
        ----------
        name : 如果是单个列名, 只能是字符串类型。如果是多列，可以是任意 iterable。"""
        if isinstance(name, str):
            self.__drop.append(name)  # 如果是单个列。单列只能是字符串，不能是其他类型
        else:
            self.__drop.extend(name)

    def remove_drop(self, name):
        """把列名 name 从已剔除集合中移除。\n
        参数:
        ----------
        name : 需要删除的列名。如果是单个列名, 只能是字符串类型。如果是多列，可以是任意 iterable。"""
        if isinstance(name, str):
            self.__drop.remove(name)
        else:
            for namei in name: self.drop.remove(namei)

    def add(self, name, key='num'):
        """添加 cate|num|ordi 类型的列名。函数无返回值 \n
        参数:
        ----------
        name : 需要添加的列名。如果是单个列名, 只能是字符串类型。如果是多列，可以是任意 iterable。\n
        key: str, 把name列添加到哪个类别中去，可选值有'cate','num','ordi'，传入其他值会报错。"""
        assert key in ('cate', 'num', 'ordi'), "key not in (cate,num,ordi)"
        single_col = isinstance(name, str)
        if key == 'cate':
            self.__cate.append(name) if single_col else self.__cate.extend(name)
        elif key == 'num':
            self.__num.append(name) if single_col else self.__num.extend(name)
        else:
            self.__ordi.append(name) if single_col else self.__ordi.extend(name)

    def __str__(self):
        n = {'cate': len(self.cate()), 'num': len(self.num()),
             'ordi': len(self.ordi()), 'drop': len(self.drop())}
        a = []
        for i in ('cate', 'ordi', 'num'):
            a.append('%-4s 个数: %4d' % (i, n[i]))
        a = '\n'.join(a)
        return a

    def __repr__(self):
        return self.__str__()

    @property
    def detail(self):
        """打印出各类变量的详细清单"""
        d = {'cate': self.cate(), 'num': self.num(),
             'ordi': self.ordi(), 'drop': self.drop()}
        for key in ('cate', 'ordi', 'num', 'drop'):
            print(key, ":")
            for col in d[key]:
                print("    {},".format(col))


def cal_woe(tab_df, sort_key='woe'):
    """根据分箱变量和目标变量的列联表，计算 woe, iv, gini。tab_df 应包含分箱变量的所有箱。  \n
    本模块把这个函数计算的结果表，称作woe_df。很多函数的返回值，都是woe_df。  \n
    参数:
    ----------
    tab_df：分箱变量和目标变量的列联表   \n
    sort_key: 'woe' or 'index', 计算 Gini 时按 woe 排序还是分箱的值排序。 \n
    返回值:
    ---------
    woe_df: dataframe, 包含的列及其解释如下 \n
        colName: 列名, \n
        0: good, \n
        1: bad, \n
        All: 行汇总(good + bad), \n
        binPct: All 的列占比（All_i / All.sum()）, \n
        badRate: 行 bad 占比（每一行中，bad/All \n
        woe: 分箱的 woe 值, \n
        IV_i: 分箱的 IV 值, \n
        IV: 变量的 IV 值, \n
        Gini: 变量的 Gini 系数值
    """
    woe_df = tab_df.copy()
    woe_df['All'] = woe_df[1] + woe_df[0]
    dfsum = woe_df.sum()
    woe_df['binPct'] = woe_df['All'] / dfsum.All
    woe_df['badRate'] = woe_df[1] / woe_df['All']

    good_pct = woe_df[0].values / dfsum[0]
    bad_pct = woe_df[1].values / dfsum[1]
    while not bad_pct.all():
        i = bad_pct.argmin()
        bad_pct[i] = 1 / dfsum[1]  # 若箱内 bad 为 0，则用 1 代替计算 bad_i / bad_all
    woe_df['woe'] = np.log(good_pct / bad_pct)
    woe_df['IV_i'] = (good_pct - bad_pct) * woe_df['woe']
    woe_df['IV'] = woe_df['IV_i'].sum()

    if sort_key == 'woe':
        dfGini = woe_df.sort_values(by='woe')  # 计算 gini 系数前，需要排序.
    else:
        dfGini = woe_df.sort_index()
    dfCum = pd.DataFrame([[0, 0]], columns=[0, 1])
    dfCum = dfCum.append(dfGini[[0, 1]].cumsum() / dfsum.loc[[0, 1]])  # 累计占比
    area = 0
    for i in range(1, len(dfCum)):
        area += 1 / 2 * (dfCum[1].iloc[i - 1] + dfCum[1].iloc[i]) * (dfCum[0].iloc[i] - dfCum[0].iloc[i - 1])
    woe_df['Gini'] = 2 * (area - 0.5)
    return woe_df


def crosstab_na(index, column):
    """求列联表时，把缺失也统计进来. pandas 自带的 crosstab 会把缺失值忽略掉  \n
    参数:
    ----------
    index : series, 用作列联表行维的series  \n
    column : series, 用作列联表列维的series, 应与 index 列等长   \n
    返回值:
    ----------
    tab_df: dataframe, 两个输入列的列联表，列中的缺失值也会统计进来"""
    tab_df = pd.crosstab(index, column)

    logic = index.isnull()
    if logic.any():
        na_row = column[logic].value_counts().to_frame(name=np.nan).transpose()
        tab_df = pd.concat([na_row, tab_df])

    logic = column.isnull()
    if logic.any():
        na_col = index[logic].value_counts(dropna=False).to_frame(name=np.nan)
        tab_df = pd.concat([tab_df, na_col], axis=1)

    return tab_df


def crosstab_pct(index, column, dropna=False):
    """把列联表的个数及占比均求出来 \n
    参数:
    ----------
    index : series, 用作交叉表行维的列  \n
    column : series, 用作交叉表列维的列  \n
    dropna : bool, 是否要把缺失值丢弃 \n
    返回值:
    ---------
    tab_with_pct: dataframe, 其中列联表个数在上半部分，列联表的占比在下半部分
    """
    if dropna:
        a = pd.crosstab(index, column, margins=True)
    else:
        a = crosstab_na(index, column)
        a['All'] = a.sum(axis=1)
        a.loc['All'] = a.sum()
    b = a / a.iloc[-1, -1]
    return pd.concat([a, b])


def cross_woe(var, y, dropna=False):
    """以分箱变量var和目标变量y，计算woe_df。\n
    参数:
    ----------
    var: sr, 分好箱的离散特征变量   \n
    y: sr, 模型的目标变量  \n
    dropna: 是否要丢弃 var 变量的 nan 分组，默认为False \n
    返回值:
    ---------
    woe_df: dataframe, 包含的列及其解释如下 \n
        colName: 列名, 0: good, 1: bad, All: 行汇总(good+bad), binPct: All列的列占比, badRate: 行bad占比 \n
        woe: 分箱的woe值, IV_i: 分箱的IV值, IV: 变量的IV值, Gini: 变量的Gini系数值  \n
    参见 cal_woe, cross_woe 会调用 cal_woe 函数。"""
    var, y = na_split_y(var, y)
    if not dropna:
        tab_df = crosstab_na(var, y)
    else:
        tab_df = pd.crosstab(var, y)

    try:
        tab_df['colName'] = var.name
    except:
        tab_df['colName'] = 'var'
    return cal_woe(tab_df.reindex(columns=['colName', 0, 1]))


def cross_woe_batch(detail_df, y_name, cols=None, dropna=True):
    """对数据表 detail_df 中的所有在 cols 中的列，批量计算 woe_df  \n
    参数:
    ----------
    detail_df: 明细数据表 \n
    y_name: str, detail_df 中目标变量的名字 \n
    cols: str or sequence_of_str, 需要计算 woe_df 的列的集合, 这些列应是分好箱的列. None表示所有列 \n
    dropna: 是否要丢弃 var 变量的 nan 分组，默认为False \n
    返回值:
    ---------
    woe_df: dataframe, 不同于cross_woe, 此函数返回的 woe_df 包含了多个变量。 \n
    参见 cross_woe
    """
    if cols is None:
        cols = detail_df.columns.drop(y_name)
    if isinstance(cols, str):
        cols = [cols]  # str 表示单个列

    woe_df = pd.DataFrame()
    for i in cols:
        tmp = cross_woe(detail_df[i], detail_df[y_name], dropna)
        woe_df = woe_df.append(tmp)
    return woe_df


def binning(score, bins):
    """根据 bins 把 score 离散化分组，分数最低的在第 1 组，最高的在 len(bins)+1 组.\n
    如果 score 有缺失，缺失被分在第 0 组。区间均为左闭右开型：[low,high) \n
    参数:
    ----------
    score: Series or 1darray, 需要离散化的任意变量，支持排序即可  \n
    bins: list, 对score进行离散化的分割点。n 个点会把 score 离散化成 n+1 个组, bins 的分割点可以乱序 \n
    返回值:
    ----------
    group: 1darray, 离散化后的一维数组"""
    score = np.array(score)
    group = np.zeros(len(score))
    bins = sorted(bins)
    logic0 = np.array([False] * len(score))
    n = 1
    for i in bins:
        logic1 = score < i  # 分组区间左包含右不包含
        group[~logic0 & logic1] = n
        logic0 = logic1
        n += 1
    logic1 = score >= i  # 如果 bins 为空列表，此处会报错说i未赋值
    group[~logic0 & logic1] = n
    n = len(bins)
    if n == 1:
        label_name = [str(bins[0]) + '-', str(bins[0]) + '+']
    else:
        label_name = ['{0}~{1}'.format(bins[i], bins[i + 1]) for i in range(n - 1)]
        label_name = [str(bins[0]) + '-'] + label_name + [str(bins[-1]) + '+']
    label_name = {i: str(i) + '_' + j for i, j in zip(range(1, n + 2), label_name)}
    label_name[0] = '0_nan'  # 缺失分组为'nan'
    group = np.array([label_name.get(i) for i in group])
    return group


def monotony_type(woe_sr):
    """判断分箱变量的 woe 的单调类型：单调上升、单调下降、U形、倒U形  \n
    参数:
    ----------
    woe_sr: 分箱变量的各个箱的woe。sr 应按箱的业务逻辑排好序了再传给此函数。  \n
    返回值:
    ----------
    mono: str, 可能的取值有('单升', '单降, 'U形', '倒U', '无序')  """

    def arising(sr):
        """单调上升"""
        diff = sr.diff() > 0
        return diff[1:].all()

    def derising(sr):
        """单调下降"""
        return arising(-sr)

    def u_like(sr):
        """U形"""
        idx = sr.argmin()
        left = sr[:idx]
        right = sr[idx:]
        if len(left) == 0 or len(right) == 0: return False
        return derising(left) and arising(right)

    def n_like(sr):
        """倒U形"""
        return u_like(-sr)

    if arising(woe_sr):
        mono = '单升'
    elif derising(woe_sr):
        mono = '单降'
    elif u_like(woe_sr):
        mono = 'U形'
    elif n_like(woe_sr):
        mono = '倒U'
    else:
        mono = '无序'
    return mono


def join_row(tab_df, n, iadd=1):
    """把第 n 行与其他行相加合并，iadd一般为1或-1，即相加相邻行到第 n 行，同时删除相邻行 \n
    合并分箱时，经常会用到此函数。
    参数:
    ----------
    tab_df : 需要操作的数据表\n
    n : 需要合并的行的序号\n
    iadd: 相对于n的行号偏移量。\n
    返回值:
    ----------
    new_df: 新的 dataframe, 不修改原 df"""
    tab_df = tab_df.copy()
    tab_df.iloc[n, :] = tab_df.iloc[n, :] + tab_df.iloc[n + iadd, :]
    return tab_df.idrop_axis(n + iadd)


def join_k(tab_df, k=100):
    """样本量小于 k 的箱，统一合并到other组中。\n
    参数:
    ----------
    tab_df : 列联表，表的每一行是一个箱 \n
    k: int, 样本量的阈值
    返回值:
    ----------
    tab_k: dataframe, 相比输入参数tab, 删除了样本量小于 k 的行，并增加了一个 "other" 箱
    """
    All = tab_df[1] + tab_df[0]
    logic = All < k
    other = tab_df[logic].sum().to_frame(name='other').transpose()
    other[[0, 1]] = other[[0, 1]].astype(tab_df[1].dtype)
    tab_k = tab_df[~logic]
    if logic.any():
        tab_k = pd.concat([other, tab_k])
    return tab_k


def join_nn(tab_df, tol=0.005):
    """badRate 最接近的两个相邻组最先被合并，如此反复直到所有组的 badRate 差异均大于阈值 \n
    最简单、原始的箱合并算法 \n
    参数:
    ----------
    tab_df: 有badRate列的列联表 \n
    tol: badRate差异的阈值 \n
    返回值:
    ----------
    tab_nn: dataframe, 合并了所有相邻箱中，badRate 差异小于阈值的组"""
    diff = tab_df['badRate'].diff().abs()
    min_diff = diff.min()
    if min_diff > tol or len(tab_df) == 1:
        return tab_df
    else:
        idx = diff.idxmin()
        i = diff.index.find(idx)
        tab_df = join_row(tab_df, i, -1)
        tab_df['badRate'] = tab_df[1] / (tab_df[0] + tab_df[1])
        return join_nn(tab_df, tol=tol)


def ord2int(detail_df, ord_dict):
    """把序数型变量转换为int型，int的大小代表了序数变量的排序 \n
    参数:
    ----------
    detail_df: dataframe, 明细数据表 \n
    ord_dict: dict，以列名为键，以按序排列的类别值组成的list为值\n
    返回值：
    ----------
    ord_df: 转换成int型的明细数据表, 后缀'_ord'的列是映射成的 int 型列 \n
    ord_map: dict, 各列的转换映射, 即每一列中，{类别值: 序号} 的映射字典"""
    ord_df = pd.DataFrame(index=detail_df.index)
    ord_map = {}
    for col, value in ord_dict.items():
        ord_map[col] = {i: num for i, num in zip(value, range(1, len(value) + 1))}
        ord_df[col + '_ord'] = detail_df[col].map(lambda x: ord_map[col][x])
    return ord_df, ord_map


def cate2int(detail_df, cols=None):
    """把类别型变量的类别值，映射成integer.\n
    参数:
    ----------
    detail_df: dataframe, 包含了类别型变量的明细数据框 \n
    cols: array-like, 类别型变量名组成的列表，None 表示 df_cate 中所有列都是类别型变量，需要映射。\n
    返回值：
    ----------
    cate_df: dataframe, 对每一列，其带后缀'_cate'的列是映射的int型列 \n
    cate_map: 映射字典， 即每一列中，{类别值: 序号} 的映射字典"""
    if cols is None:
        cols = detail_df.columns
    if isinstance(cols, str):
        cols = [cols]

    cate_df = pd.DataFrame(index=detail_df.index)
    cate_map = {}
    for col in cols:
        cates = detail_df[col].unique()
        cate_map[col] = dict(zip(cates, range(len(cates))))
        cate_df[col + '_cate'] = detail_df[col].map(lambda x: cate_map[col].get(x))
    return cate_df, cate_map


def cate2onehot(detail_df, cols=None, sparse=False):
    """把类别型变量的类别值，转换成独热编码，并标准化.\n
    参数:
    ----------
    detail_df: dataframe, 包含了类别型变量的明细数据框 \n
    cols: array-like, 类别型变量名组成的列表，None 表示 df_cate 中所有列都是类别型变量，需要映射。\n
    返回值：
    ----------
    x_cate_df: dataframe, 对每一列，其带后缀'_onehot'的列是映射的onehot编码  \n
    cate_imap: 逆映射字典，即每一列中，{序号:类别值} 的映射字典 """
    from sklearn.preprocessing import OneHotEncoder
    cate_df, cate_map = cate2int(detail_df, cols)
    encoder = OneHotEncoder(sparse=sparse)
    encoder.fit(cate_df)
    x_cate_onehot = encoder.transform(cate_df)
    x_cate_std = x_cate_onehot * np.sqrt(0.5)

    cols_onehot = [i[:-4] + 'onehot' for i in cate_df]  # 把 col+'_cate' 变量名改成 col+'_onehot' 变量名
    cols_onehot_expand = []   # 扩展成大宽表后，x_cate_std 中各列的名字
    for col, unique_cate in zip(cols_onehot, encoder.n_values_):
        cols_onehot_expand.extend(col + str(i) for i in range(unique_cate))
    x_cate_df = pd.DataFrame(x_cate_std, index=detail_df.index, columns=cols_onehot_expand)
    cate_imap = {col: tools.reverse_dict(value) for col, value in cate_map.items()}
    return x_cate_df, cate_imap

class NumBin:
    """连续型/序数型变量的分箱，使用的是决策树算法。\n
    Parameters
    ----------
    所有传入的参数都会传给 sklearn.tree.DecisionTreeClassifier 以生成实例。 \n
    criterion : str(default='gini')
        使用哪个指标来寻找最佳分割点，可选择 'gini' 或 'entropy' 作为分箱指标。
    max_leaf_nodes : int(default=8)
        最大叶节点个数，设置此参数后，决策树将按深度优先模式生长。
    min_samples_leaf : int(default=300)
        最小叶节点样本容量。"""

    def __init__(self, criterion='gini', max_leaf_nodes=8,
                 min_samples_leaf=300, **kwargs):
        self.kwargs = kwargs
        self.kwargs['criterion'] = criterion
        self.kwargs['max_leaf_nodes'] = max_leaf_nodes
        self.kwargs['min_samples_leaf'] = min_samples_leaf

    def fit(self, sr, y, max_leaf_nodes=None, min_samples_leaf=None):
        """数值型/序数型变量的分箱，按决策树算法寻找最优分割点。\n
        Parameters:
        ----------
        sr : Series
            数值型预测变量。
        y : Series or ndarray
            目标变量，取值 {0, 1}
        max_leaf_nodes : int(default=None)
            如果有值，则使用传入值。否则使用初始化时的值
        min_samples_leaf : int(default=None)
            如果有值，则使用传入值。否则使用初始化时的值
        返回值:
        ----------
        无返回值，但会更新 self 对象的如下属性：\n
            clf_: 训练好的决策树分类器 \n
            bins_: 算法找出的最佳分割点 \n
            tab_: 根据最佳分割点计算出来的 woe_df  \n
            name_: 训练的输入 series 的 name 属性值，即特征变量的名字"""

        def tree_split_point(fitted_tree):
            """解析并返回树的各个分割节点"""
            bins = sorted(set(fitted_tree.tree_.threshold))
            return [round(i, 4) for i in bins]

        kwargs = self.kwargs.copy()
        if max_leaf_nodes is not None:
            kwargs['max_leaf_nodes'] = max_leaf_nodes
        if min_samples_leaf is not None:
            kwargs['min_samples_leaf'] = min_samples_leaf
        self.clf_ = DecisionTreeClassifier(**kwargs)
        na_group, var_no, y_no = na_split(sr, y)
        self.clf_.fit(var_no.to_frame(), y_no)
        bins = tree_split_point(self.clf_)
        self.bins_ = bins
        self.name_ = sr.name
        binvar = self.transform(sr)  # 包含有对缺失值的处理
        self.tab_ = cross_woe(binvar, y)

    def transform(self, sr):
        """根据训练好的最优分割点，把原始变量转换成分箱变量 \n
        参数:
        ------------
        sr: series, 原始变量，与训练时的变量含义相同时，做此转换才有意义  \n
        返回值:
        ------------
        binvar: ndarray, 转换成分箱变量，离散化成了有限组的一维数组"""
        binvar = binning(sr, self.bins_)
        return binvar


class CateBin:
    """
    类别型变量的分箱，用卡方检验把相近的箱合并成一个箱。\n
    Parameters
    ----------
    tol : float, optional(default=0.1)
        卡方独立性检验的 p 值阈值，大于等于 tol 的相邻组会合并。p越小，最终分组间的
        差异越显著。可选值有4个：0.15、0.1、0.05、0.01\n
    k : int, optional(default=100)
        样本容量的阈值，小于 k 的所有类别被归入 "other" 类。"""

    def __init__(self, tol=0.1, k=100):
        # 四个置信度对应的卡方值。卡方值小于阈值的组需要合并
        chi2_level = {0.15: 2.0768, 0.1: 2.7055, 0.05: 3.8596, 0.01: 6.6667}
        self.tol = tol
        self.chi2_tol = chi2_level[tol]
        self.k = k

    @staticmethod
    def cal_chi(tab_df):
        """四格列联表计算卡方值的公式"""
        a = tab_df.iloc[0, 0]
        b = tab_df.iloc[0, 1]
        c = tab_df.iloc[1, 0]
        d = tab_df.iloc[1, 1]
        chi2 = (a * d - b * c)**2 * (a + b + c + d) / ((a + b) * (c + d) * (a + c) * (b + d))
        return chi2

    @staticmethod
    def cross_var(sr, y):
        """对离散变量做交叉列联表\n
        参数:
        ----------
        sr: 取值为离散值的Series，一般是分了箱的一个特征 \n
        y: 目标变量 \n"""
        tab_df = crosstab_na(sr, y)
        tab_df['badRate'] = tab_df[1] / (tab_df[0] + tab_df[1])
        tab_df['bins'] = [[i] for i in tab_df.index]  # bins用来记录分组合并的细节
        tab_df = tab_df.sort_values(by='badRate', ascending=0)
        return tab_df

    def fit(self, sr, y, tol=None, k=None):
        """
        类别型变量的分箱。采用的是卡方合并法，2个类别的卡方独立检验值小于阈值时，合并它们。\n
        Parameters
        ----------
        sr : Series or iterable
            分箱的自变量，一般是Series。
        y : Series or iterable
            分箱的目标变量，必须与 sr 等长。
        tol : float
            卡方独立性检验的 p 值阈值，大于等于 tol 的相邻组会合并。若 tol 为 None，则使用实例化时的 tol。\n
            p越小，最终得到的分组间的差异越显著。可选值有4个：0.15、0.1、0.05、0.01
        k : int
            样本容量的阈值，小于 k 的所有类别被归入 "other" 类。若k为None，则使用实例化时的k。
        返回值:
        ----------
        无返回值，但会更新 self 对象的如下属性：\n
            bins_: 算法找出的合并分箱的逻辑 \n
            tab_: 根据最佳合并分箱计算出来的 woe_df  \n
            name_: 训练的输入 series 的 name 属性值，即特征变量的名字"""

        if tol is None:
            tol = self.tol
        if k is None:
            k = self.k
        chi2_level = {0.15: 2.073, 0.1: 2.69, 0.05: 3.8596, 0.01: 6.6667}
        chi2_tol = chi2_level[tol]
        tab = self.cross_var(sr, y)
        tab = join_k(tab, k=k)
        if 'other' in tab.index:
            ind = tab.index.drop('other')
        else:
            ind = tab.index

        # 计算两两类别组合而成的四格列联表的卡方值     
        # tol_df保存计算好的卡方值，每次递归都检查和修改它的值
        tol_df = pd.DataFrame(index=ind, columns=ind)
        for i in range(1, len(tol_df)):
            for j in range(i):
                idx_i = ind[i]
                idx_j = ind[j]
                tol_df.loc[idx_i, idx_j] = self.cal_chi(tab.loc[[idx_i, idx_j]])

        def min_chi2(tab, tol_df, loop_i):
            """合并卡方值最小的2个组，直到所有组的卡方值都大于等于阈值，或除了other组外只剩2个组"""
            minc = tol_df.min().min()
            if minc >= chi2_tol or len(tol_df) == 2:
                return tab
            else:
                logic = tol_df.min() == minc
                idx_j = logic.index[logic][0]
                logic = tol_df.min(axis=1) == minc
                idx_i = logic.index[logic][0]

                label = 'merge_' + str(loop_i)
                tab.loc[label, :] = tab.loc[idx_j] + tab.loc[idx_i]
                tab = tab.drop([idx_i, idx_j], axis=0)
                tol_df = tol_df.drop([idx_i, idx_j], axis=0)
                tol_df = tol_df.drop([idx_i, idx_j], axis=1)

                for idx in tol_df:
                    tol_df.loc[label, idx] = self.cal_chi(tab.loc[[label, idx]])
                tol_df[label] = np.nan

                return min_chi2(tab, tol_df, loop_i + 1)

        tab = min_chi2(tab, tol_df, 1)
        tab['colName'] = sr.name
        tab = cal_woe(tab.reindex(columns=['colName', 0, 1, 'bins']))
        col = list(tab.columns.drop('bins')) + ['bins']
        self.tab_ = tab[col]
        self.bins_ = tab['bins']
        self.name_ = sr.name

    def transform(self, sr):
        """根据训练好的分箱逻辑，把分类变量转换成分箱变量. \n
        参数:
        ------------
        sr: series, 原始变量，与训练时的变量含义相同时，做此转换才有意义  \n
        返回值:
        ------------
        binvar: series, 转换成的分箱变量"""

        def trans(x):
            for label, bin_list in zip(self.bins_.index, self.bins_):
                if pd.isnull(x):  # x 是缺失时，x in bin_list 并不能判断出真实结果
                    if pd.isnull(label):
                        return '0_nan'
                    else:
                        if any(pd.isnull(i) for i in bin_list):
                            return label
                else:
                    if x in bin_list:
                        return label

        return sr.map(trans)

    def generate_transform_fun(self):
        """生成转换函数的代码"""
        fun_code = """
        def {varname}_trans(x):
            # {varname} 分类特征的分箱转换函数
            labels = {index}
            bins = {bins}
            for label, bin_list in zip(labels, bins):
                if pd.isnull(x):  # x 是缺失时，x in bin_list 并不能判断出真实结果
                    if pd.isnull(label):
                        return '0_nan'
                    else:
                        if any(pd.isnull(i) for i in bin_list):
                            return label
                else:
                    if x in bin_list:
                        return label""".format(varname=self.name_,
                                               index=list(self.bins_.index),
                                               bins=list(self.bins_))
        return fun_code


def var_rank(woe_df, by='Gini'):
    """根据分箱变量结果，按 Gini/IV 降序排列，返回各变量的 Gini 和 IV \n
    参数:
    ----------
    woe_df: dataframe, 包含多个分箱变量的 woe_df \n
    by: str, 可选值有 ('Gini', 'IV'), 按哪一列降序排序 \n
    返回值:
    ----------
    woe_rank: dataframe, 排序后的表，包含colName, IV, Gini 三列"""
    df_dup = woe_df.drop_duplicates('colName').sort_values(by, ascending=False)
    return df_dup[['colName', 'IV', 'Gini']]


def var_filter(woe_df, k_iv=0.02, k_gini=0.04):
    """ 把 IV、Gini 均小于阈值的变量剔除，返回大于阈值的变量  \n
    参数:
    ----------
    woe_df: dataframe, 包含多个分箱变量的 woe_df \n
    k_iv: float, 变量IV的阈值 \n
    k_gini: float, 变量Gini的阈值 \n
    返回值:
    ----------
    var_filtered: dataframe, 剔除了IV和Gini均小于阈值的变量，包含colName, IV, Gini 三列"""
    rank = var_rank(woe_df)
    logic = (rank.IV >= k_iv) & (k_gini >= k_gini)
    return rank[logic].colName.tolist()


def corr_filter(detail_df, iv_dict, tol=0.8):
    """相关性系数大于等于参数tol的列，将其中IV值更低的列删除 \n
    参数:
    ----------
    detail_df: dataframe, 需要计算相关性的明细数据框 \n
    iv_dict: dict or series, 各个变量的IV或Gini指标 \n
    tol: float, 线性相关性阈值 \n
    返回值:
    ----------
    corr_df: dataframe, 相关性矩阵，并删除了相关性超过阈值的列 \n
    dropped_col: list, 删除的列"""
    corr_df = detail_df.corr_tri()
    corr_df = corr_df.abs()
    dropped_col = []
    while True:
        row, col = corr_df.argmax()
        if corr_df.loc[row, col] >= tol:
            drop_label = row if iv_dict[row] < iv_dict[col] else col
            dropped_col.append(drop_label)
            corr_df = corr_df.drop(drop_label).drop(drop_label, axis=1)
            if len(corr_df) == 1:
                break
        else:
            break
    return corr_df, dropped_col


def woe_dict(woe_df, woe_col='woe'):
    """根据分箱列联表woe_df，生成 woe_map 字典，把每个变量的分箱值映射成woe值 \n
    若 woe_df 表中有各分箱的评分，此函数也可以生成 score_map 字典，即各个变量的 {分箱值:得分值} 字典  \n
    Parameters
    ----------
    woe_df: dataframe, 包含多个分箱变量的 woe_df  \n
    woe_col : str
        woe值所在的列名
    返回值:
    ----------
    woe_map: dict, 以列名为key, 以{分箱值: woe值} 字典为值。此映射字典用来把明细数据框中分箱变量转换成woe变量"""
    cols = woe_df.colName.unique()
    woe_map = {}
    for i in cols:
        df_i = woe_df[woe_df.colName == i]
        woe_map[i] = {key: val for key, val in zip(df_i[woe_col], df_i[woe_col])}
    return woe_map


def bin2woe(woe_map, detail_df, postfix='woe'):
    """将分箱转换成woe值，给detail_df 增加上woe变量。 \n
    函数没有返回值，会修改 detail_df。
    参数:
    -----------
    woe_map: dict, 以列名为key, 以{分箱值: woe值} 字典为值。此字典一般由 woe_dict 函数返回 \n
    detail_df: dataframe, 明细数据表，包含有以 woe_map 中的key为列名的各个分箱变量 \n
    postfix: str, 此函数会修改 detail_df, 为其添加上转换的 woe 变量。此参数含义是 \n
        woe变量名 = 分箱变量名[:-3] + '_' + postfix，即为原变量名添加后缀"""
    for bin_col in woe_map:
        col_woe_dict = woe_map[bin_col]
        tmp = pd.Series(np.empty(len(detail_df)), index=detail_df.index)
        for key, val in col_woe_dict.items():
            logic = detail_df[bin_col] == key
            tmp[logic] = val
        detail_df[bin_col[:-3] + postfix] = tmp


def bin2woe_onekey(detail_df, bin_cols, y_name='fspd10', postfix='woe'):
    """求出woe_df, 同时给 detail_df 添加上“由分箱变量转换而来的woe变量”。\n
    参数:
    -----------
    detail_df: dataframe, 包含了分箱变量的明细数据框 \n
    bin_cols: iterable, 分箱变量名组成的可迭代对象  \n 
    y_name: str, 目标变量的名字  \n
    postfix: str, 此函数会修改 detail_df, 为其添加上转换的 woe 变量。此参数含义是 \n
        woe变量名 = 分箱变量名[:-3] + '_' + postfix，即为原变量名添加后缀 \n
    返回值:
    ------------
    woe_map: dict, 以列名为key, 以{分箱值: woe值} 字典为值。 \n
    woe_df: dataframe，各分箱变量的woe_df"""
    woe_df = pd.DataFrame()
    woe_map = {}
    for i in bin_cols:
        tmp = cross_woe(detail_df[i], detail_df[y_name])
        woe_map[i] = {key: val for key, val in zip(tmp.index, tmp.woe)}
        woe_df = woe_df.append(tmp)
    bin2woe(woe_map, detail_df, postfix=postfix)
    return woe_map, woe_df


def generate_scorecard(betas, woe_df, A=427.01, B=-57.7078):
    """根据逻辑回归的系数，把woe值转换成各箱的得分，输出标准评分卡。  \n
    Parameters
    ----------
    betas : series or dict, 逻辑回归得到的各变量的系数, 应包括常数项 Intecept 或 const  \n
    woe_df : dataframe, 各变量的分箱数据表   \n
    A: float, 分数转换的参数A   \n
    B: float, 分数转换的参数B   \n
    Returns
    ----------
    scorecard_df : dataframe, 标准的评分卡"""
    if isinstance(betas, dict):
        betas = pd.Series(betas)
    betas = betas.rename_axis({'Intercept': 'const'})
    const_df = pd.DataFrame({'colName': 'const', 'woe': 1})
    scorecard_df = pd.concat([const_df, woe_df])
    betas = betas.reset_index().rename(columns={'level_0': 'colName', 0: 'betas'})
    scorecard_df = scorecard_df.merge(betas, on='colName')
    scorecard_df['score'] = B * scorecard_df['betas'] * scorecard_df['woe']  # 各变量的各箱得分

    base_score = A + B * betas['const']  # 基础分
    scorecard_df.loc[scorecard_df.colName == 'const', 'score'] = base_score
    return scorecard_df


def bin2score(score_map, detail_df, version='', postfix='scr'):
    """把各变量的分箱映射成得分值。  \n
    函数无返回值，会原地修改detail_df，增加各变量得分列和总分列。  \n
    参数:
    -------------
    score_map: dict, 以分箱变量名为key， 以 {分箱值:得分值} 为值。score_map可以通过 woe_dict 函数得到 \n
    detail_df: dataframe, 明细数据框，包含有所有分箱变量 \n
    version: str, 评分卡的版本。一般一张评分卡会不定期迭代，每次迭代应有一个版本号 \n
    postfix: str, 此函数会修改 detail_df, 为其添加上转换的 打分变量。此参数含义是 \n
        打分变量名 = 分箱变量名[:-3] + '_' + postfix，即为原变量名添加后缀 \n
    """
    score_map = score_map.copy()
    try:
        const = score_map.pop('const')
    except KeyError:
        const = score_map.pop('Intercept')

    score_fun = lambda x: score_map[col].get(x, -9999)  # 字典未覆盖的分箱值，其得分为-9999，一般表示数据出现清洗错误
    col_list = []
    for col in score_map:
        col_score = col[:-3] + postfix
        detail_df[col_score] = detail_df[col].map(score_fun)
        col_list.append(col_score)

    total_score = 'score_' + version if version else 'score'
    detail_df[total_score] = detail_df[col_list].sum(axis=1) + const

    # 检查有无打分异常的变量
    err = detail_df[col_list].min()
    logic = err == -9999
    if logic.any():
        print("以下变量的打分有异常值:")
        print(err[logic].index.tolist())


def psi(pct_base, pct_actual):
    """计算变量的稳定性指标 psi。\n
    参数:
    -------------
    pct_base : series, 是基准分布。 \n
    pct_actual : series or dataframe, 是要比较的分布。如果是df，则每一列应该是一个分布。pct_actual的index应与pct_base的index对齐  \n
    返回值:
    -------------
    psi_df: dataframe，带psi结尾的列是各列相对于基准的psi"""
    psi_df = pd.concat([pct_base, pct_actual], axis=1)
    psi_df = psi_df / psi_df.sum()
    base = psi_df.iloc[:, 0]
    col_compare = psi_df.columns[1:]
    for col in col_compare:
        psi_i = (psi_df[col] - psi_df[base]) * np.log(psi_df[col] / psi_df[base])
        psi_df[str(col) + '_psi'] = psi_i.sum()
    return psi_df


def vsi(pct_base, pct_actual):
    """计算变量的分数迁移性指标。\n
    参数:
    -------------
    pct_base : dataframe, 包含2列：binPct列是基准分布，score列是基准得分。   \n
    pct_actual : sr or df, 是要比较的分布。如果是df，则每一列应该是一个分布。pct_actual的index应与pct_base的index对齐  \n
    返回值:
    -------------
    vsi_df: dataframe，带vsi结尾的列是各列的分数迁移"""
    vsi_df = pd.concat([pct_base, pct_actual], axis=1)
    base_pct = vsi_df['binPct']
    col_compare = vsi_df.columns[2:]
    vsi_df[col_compare] = vsi_df[col_compare] / vsi_df[col_compare].sum()
    for col in col_compare:
        shifti = (vsi_df[col] - base_pct) * vsi_df['score']
        vsi_df[str(col) + '_vsi'] = shifti.sum()
    return vsi_df


def woe_trend(detail_df, bin_cols, by, y="fpd10"):
    """在时间维度上看变量的woe变化趋势，即每个周期内都计算一次分箱变量的woe值，按周期看woe值的波动情况 \n
    参数:
    ---------
    detail_df: 明细数据表，至少应包含分箱变量、时间变量、目标变量  \n
    bin_cols: iterable, 分箱变量  \n
    by: str, 时间维度的变量名，detail_df将按此变量的值做groupby，每一组内计算一次各变量的woe值。  \n
    y: str, 目标变量的名字
    返回值:
    ---------
    woe_period: dataframe, columns是时间的各个周期，row是各个分箱变量的分箱值，values是各个箱在各个周期内的woe值
    """
    group = detail_df.groupby(by)
    badRate = group[y].mean()  # 各周期内的全局 badrate, 是各周期内计算woe值的基准
    df_woe = []
    for var in bin_cols:
        var_badrate = group.apply(lambda x: x.pivot_table('fpd10', var, aggfunc='mean')).transpose()
        for j in var_badrate:
            badrate_base = badRate[j]
            var_badrate[str(j) + '_woe'] = var_badrate[j].apply(bad2woe, args=(badrate_base,))
        var_badrate['colName'] = var
        df_woe.append(var_badrate)
    return pd.concat(df_woe)


def matrix_rule(matrix_df, row_name, col_name):
    """矩阵细分：分别以2个变量的取值为行、列索引，在matrix_df矩阵中查找值，此值便是矩阵细分的返回值。 \n
    参数:
    ---------
    matrix_df: dataframe, 交叉规则矩阵。index，column代表输入变量的取值条件，value代表此条件下对应的输出。 \n
        此矩阵表达了矩阵细分的逻辑
    row_name: str, 行方向的输入变量的名字，row 参数是输入条件之一 \n
    col_name: str, 列方向的输入变量的名字，col 参数是输入条件之二 \n
    返回值:
    ---------
    apply_fun: 函数，以series为输入，以 matrix_df的values中的值为返回值。\n
        函数用法：df.apply(apply_fun, axis=1), 即以 df 的各行 series 作为 apply_fun 的输入参数。 \n
    示例:
    ----------
    matrix_df = pd.DataFrame(data=[[1,2],[3,4]], index=['Male', 'Female'], columns=['Married', 'UnMarried']) \n
    detail_df = pd.DataFrame({'gender':['Male', 'Female','Male', 'Female','Male', 'Female','Male', 'Female'],
                            'Marry': ['Married','Married','Married','Married','UnMarried','UnMarried','UnMarried','UnMarried']}) \n
    detail_df.apply(matrix_rule(matrix_df, 'gender', 'Marry'), axis=1)  \n
    以上代码的返回值如下： \n
    pd.Series([1,3,1,3,2,4,2,4])
    """
    def apply_fun(sr):
        index_value = sr[row_name]
        column_value = sr[col_name]
        return matrix_df.loc[index_value, column_value]
    return apply_fun


def bad2woe(badrate_i, badrate_base):
    """把 badrate 转换成 woe 值。\n
    参数:
    ---------
    badrate_i: float or 1darray or series, 各个箱的 badrate  \n
    badrate_base: float, 全局的基准 badrate  \n
    返回值:
    ----------
    woe_i: 类型同 badrate_i, 各个箱对应的 woe 值。"""
    return np.log(1 / badrate_i - 1) - np.log(1 / badrate_base - 1)


def bad2woe_base(badrate_base):
    """生成带全局基准 badrate 的转换函数，此函数能把 badrate 转换成 woe 值。\n
    参数:
    ---------
    badrate_base: float, 全局的基准 badrate  \n
    返回值:
    ----------
    bad2woe_fun: 函数，以 badrate_i（各个箱对应的badrate为输入值），返回各箱对应的 woe 值。"""
    return lambda bad_i: np.log(1 / bad_i - 1) - np.log(1 / badrate_base - 1)


def woe2bad(woe_i, badrate_base):
    """bad2woe 的逆函数，把 woe 值转换成 badrate 值。 \n
    参数:
    ---------
    woe_i: float or 1darray or series, 各个箱对应的 woe 值  \n
    badrate_base: float, 全局的基准 badrate  \n
    返回值:
    ----------
    badrate_i: 类型同 woe_i, 各个箱的 badrate。
    """
    return 1 / (1 + ((1 - badrate_base) / badrate_base) * np.exp(woe_i))


def y_transform1(y):
    """把取值为 {0, 1} 的目标变量，转换为取值为 {-1, 1} 的目标变量。SVM， AdaBoost 模型需要这样的目标变量 \n
    参数:
    ---------
    y: ndarray or series, 目标变量，其中取值 0 表示 good， 1 表示 bad  \n
    返回值:
    ----------
    y_trans: ndarray or series, 转换后的目标变量，其中取值 -1 表示 good，取值 1 表示 bad"""
    return np.array(2 * (y - 0.5))


def y_transform0(y):
    """y_transform1 的逆函数，把取值为 {-1, 1} 的目标变量，转换为取值为 {0, 1} 的目标变量。\n
    参数:
    ---------
    y: ndarray or series, 目标变量，其中取值 -1 表示 good， 1 表示 bad  \n
    返回值:
    ----------
    y_trans: ndarray or series, 转换后的目标变量，其中取值 0 表示 good，取值 1 表示 bad"""
    return np.array((y + 1) / 2)


def check_diff(sr1, sr2, detail=False):
    """检查两列的差集。拼表前检查数据质量时，经常需要对拼接键做这种检查。  \n
    此函数也可用来检查 df1, df2 的 columns 差集、df1.index 与 df2.index 差集。  \n
    参数:
    ---------
    sr1: iterable, 列1  \n
    sr2: iterable, 列2， sr1 和 sr2 是检查差集的目标对象  \n
    detail: bool, 是否要返回差集的明细 diff1, diff2 。默认只打印出差集中的元素个数，不返回差集明细  \n
    """
    diff1 = set(sr1) - set(sr2)
    diff2 = set(sr2) - set(sr1)
    print('diff1: {}'.format(len(diff1)))
    print('diff2: {}'.format(len(diff2)))
    if detail:
        return diff1, diff2


def entropy(bins):
    """计算信息熵，bins可以是各分类的频数，也可以是各分类的占比。 \n
    参数:
    ----------
    bins: float, list, series or 1darray, 各类别的频数或占比。\n
        若 bins 是 float, 则视为二分类变量中，其中一个类别的占比，比如 badrate 
    返回值:
    ----------
    entr: float, 变量的信息熵"""
    if isinstance(bins, list):
        bins = np.array(bins)
    elif isinstance(bins,float):
        bins = np.array([bins, 1-bins])
    bins = bins / bins.sum()
    entr = -bins * np.log(bins)
    entr = np.where(np.isnan(entr), 0, entr)
    return entr.sum()


def entropy_sr(sr):
    """计算信息熵，与 entropy 函数不同，此函数以一个明细的观测点序列为输入  \n
    参数:
    ----------
    sr: series, 一列明细数据，非统计好的各类别占比  \n
    返回值:
    ----------
    entr: float, 变量的信息熵"""
    p = sr.destribution()
    e = p.binPct * np.log(p.binPct)
    return -e.sum()


def cond_entropy(sr, by):
    """计算随机变量的条件熵.  \n
    参数:
    ----------
    sr: series, 一列明细数据，非统计好的各类别占比  \n
    by: series, 与 sr 等长，条件明细数据。将按 by 取不同的值分组，计算各组内 sr 的熵，再加权求和  \n
    返回值:
    ----------
    entr: float, 变量的条件熵"""
    d = by.destribution().binPct
    cond_entr = pd.Series()
    for i in d.index:
        ei = entropy_sr(sr[by == i])
        cond_entr[i] = ei
    return (cond_entr * d).sum()


def gini_impurity(bins):
    """计算基尼不纯度，以各类的频数或频率为输入。  \n
    参数:
    ----------
    bins: float, list, series or 1darray, 各类别的频数或占比。\n
        若 bins 是 float, 则视为二分类变量中，其中一个类别的占比，比如 badrate  \n
    返回值:
    ----------
    impurity: float, 变量的基尼不纯度"""
    if isinstance(bins, float):
        bins = np.array([bins, 1 - bins])
    bins = bins / bins.sum()
    impurity = 1 - (bins**2).sum()
    return impurity


def gini_impurity_sr(sr):
    """计算基尼不纯度, 以一列明细观测为输入。  \n
    参数:
    ----------
    sr: series, 一列明细数据，非统计好的各类别占比  \n
    返回值:
    ----------
    impurity: float, 变量的基尼不纯度"""
    p = sr.distribution()
    impurity = 1 - (p.binPct * p.binPct).sum()
    return impurity


def cond_gini_impurity(sr, by):
    """计算条件基尼不纯度。 \n
    参数:
    ----------
    sr: series, 一列明细数据，非统计好的各类别占比   \n
    by: series, 与 sr 等长，条件明细数据。将按 by 取不同的值分组，计算各组内 sr 的基尼，再加权求和   \n
    返回值:
    ----------
    impurity: float, 变量的条件基尼不纯度
    """
    d = by.distribution().binPct
    cond_gini = pd.Series()
    for i in d.index:
        gi = gini_impurity(sr[by == i])
        cond_gini[i] = gi
    return (cond_gini * d).sum()


def gain_ratio(sr, by, cretiria='gini'):
    """计算信息增益比. 默认以基尼不纯度为指标，来计算增益比。  \n
    参数:
    ----------
    sr: series, 一列明细数据，应是离散化的特征   \n
    by: series, 与 sr 等长，条件明细数据, 一般是目标变量。n
    cretiria: 'gini' or 'entropy', 以哪个指标来计算增益比   \n
    返回值:
    ----------
    gain: float, 基尼增益比/熵增益比。
    """
    if cretiria == 'gini':
        entr = gini_impurity_sr(sr)
        cond_entr = cond_gini_impurity(sr, by)
    elif cretiria == 'entropy':
        entr = entropy_sr(sr)
        cond_entr = cond_entropy(sr, by)
    else:
        raise Exception("cretiria not in ('gini','entropy'")
    gain = (entr - cond_entr) / entr
    return gain


def na_split(var, y):
    """把目标变量缺失值对应的数据丢弃、特征变量的缺失值对应的数据单独分离出来作为一组 \n
    参数:
    ---------
    var: series, 特征变量 \n
    y: series, 目标变量 \n
    返回值:
    ---------
    na_group: 如果var没有缺失值，np_group返回值为0；否则返回一dataframe  \n
    var_notNa: 分离缺失值后的特征变量  \n
    y_notNa: 分离缺失值后的目标变量   \n"""
    logic = pd.notnull(y)
    var = var[logic]
    y = y[logic]
    logic = pd.isnull(var)
    if logic.any():
        na_grp = pd.DataFrame(columns=[0, 1, 'All', 'badRate', 'binPct'], index=[np.nan])
        var_no = var[~logic]
        y_na, y_no = y[logic], y[~logic]
        na_grp[1] = y_na.sum()
        na_grp['All'] = len(y_na)
        na_grp[0] = na_grp['All'] - na_grp[1]
        na_grp['badRate'] = na_grp[1] / na_grp['All']
        na_grp['binPct'] = na_grp['All'] / len(y)
        return na_grp, var_no, y_no
    return 0, var, y


def na_split_y(var, y):
    """把目标变量缺失值对应的数据丢弃。很多场景下都要剥离出不可观测的数据来。  \n
    参数:
    ----------
    var: series or dataframe, 特征变量  \n
    y: series, 与 var 观测数相等，目标变量.   \n
    返回值:
    ----------
    var_notna: 类型同 var，去掉了目标变量缺失值对应的观测   \n
    y: series, 去掉了缺失值"""
    logic = pd.notnull(y)
    var_notna = var[logic]
    y_notna = y[logic]
    return var_notna, y_notna


def roc(score, y, detail=False, sample_weight=None):
    """计算gini系数，可加权重。\n
    参数:
    ---------
    score: series or ndarray, 模型的得分、概率或决策函数值，值越低表示越差。  \n
    y: series or ndarray, 模型的目标变量，取值 {0, 1}  \n
    detail: bool, 是否返回ROC曲线数据。当 detail 为 False 时只返回 gini, ks 系数;   \n
        当 detail 为 True 时返回 gini 系数和用于绘制 ROC 曲线的 fpr, tpr 数组  \n
    sample_weight: series or ndarray, 与 score 长度相等。各个观测的样本权重  \n
    返回值:
    ----------
    gini: float, 模型的基尼系数  \n
    ks: float, 模型的KS系数   \n
    fpr: 假阳率，ROC 曲线的 x 轴数据。仅当 detail 参数为 True 时返回。  \n
    tpr: 真阳率，ROC 曲线的 y 轴数据。仅当 detail 参数为 True 时返回。"""
    score, y = na_split_y(score, y)
    if sample_weight is None:
        sample_weight = np.ones(len(y))
    df = pd.DataFrame({'score': score, 'bad': y, 'weight': sample_weight})  # .sort_values('score')
    df['good_w'] = (1 - df['bad']) * df['weight']
    df['bad_w'] = df['bad'] * df['weight']
    All = np.array(df.groupby('bad')['weight'].sum())
    df_gini = df.groupby('score')[['good_w', 'bad_w']].sum().cumsum() / All

    score_min = 2 * df_gini.index[0] - df_gini.index[1]  # 比实际的最小值略小，score_min = min0 - (min1 - min0)
    df_0 = pd.DataFrame([[0, 0]], columns=['good_w', 'bad_w'], index=[score_min])
    df_gini = pd.concat([df_0, df_gini])

    A = auc(df_gini.good_w, df_gini.bad_w)
    Gini = (A - 0.5) / 0.5

    diff = df_gini['bad_w'] - df_gini['good_w']
    KS = diff.max()

    if detail:
        return Gini, KS, df_gini.good_w, df_gini.bad_w
    return Gini, KS


def gini_ks_groupby(score, y, by):
    """按指定维度分组，计算每组中的 Gini, KS 指标  \n
    参数:
    ---------
    score : series or ndarray, 模型得分、概率或决策函数值，值越低表示越差。  \n
    y : 目标变量, 0 为 good, 1 为 bad    \n
    by : series, ndarray or list_of_them . 分组的维度变量。   \n
    返回值:
    ----------
    df_metrics: dataframe, 每组的 Gini, KS 值。"""
    dfScore = pd.DataFrame({'score': score, 'y': y})

    def gini_ks(dfProb):
        d = pd.Series()
        g, k = roc(dfProb['score'], dfProb['y'])
        d['Gini'] = g
        d['KS'] = k
        return d

    df_metrics = dfScore.groupby(by).apply(gini_ks)
    return df_metrics


class ModelEval:
    """模型评估对象。用于从不同角度评估模型效果。\n
    同一个 score 和 y 的数据，可以绘制Gini图、KS图、Lift图、bins分组等。\n
    def __init__(self, score, y, score_name=None, plot=True)\n
    初始化参数:
    ------------
        score : 模型得分或概率，越高越好。 \n
        y : 模型的真实label，0表示good，1表示bad. \n
        score_name: 模型的名字，默认值 None 表示以 score.name 为名  \n
        plot : 是否在实例化时自动绘制 GiniKS 曲线图。\n
    属性说明：
    ----------
        所有属性均以'_'结尾，所有方法则以小写字母开头   \n
        score_: 模型的得分、概率或决策函数值，剔除了目标变量为缺失值的行  \n
        y_: 模型的真实目标变量，剔除了其中的缺失值   \n
        na_group_: 如果 score 中有缺失值，缺失值会单独分离成 na_group_, 以免影响 Gini, KS 值的计算结果   \n
        gini_: 模型的 gini 指数  \n
        ks_: 模型的 KS 值   \n
        ks_score_: 模型取得 KS 值时的 score 值。   \n
        good_curve_: 假阳率。信贷领域中，“假阳”就是 good，因此“假阳率”就是 good 累计占比。   \n
        bad_curve_: 真阳率。信贷领域中，“真阳”就是bad，因此“真阳率”就是 bad 累计占比。"""

    def __init__(self, score, y, score_name=None, plot=True):
        if score_name is not None:
            self.score_name_ = score_name
        else:
            try:
                self.score_name_ = score.name
            except AttributeError:
                self.score_name_ = 'train'
        self.na_group_, self.score_, self.y_ = na_split(score, y)  # 把得分缺失的样本剔除
        self.gini_, self.ks_, self.good_curve_, self.bad_curve_ = roc(self.score_, self.y_, detail=1)
        self.ks_score_ = (self.bad_curve_ - self.good_curve_).argmax()
        if plot:
            self.giniks_plot()

    def gini_plot(self, img_path=None):
        """绘制 ROC 曲线图, 并计算 Gini 系数。无返回值  \n
        参数:
        ------------
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        g, good, bad = self.gini_, self.good_curve_, self.bad_curve_
        plt.figure(figsize=(7, 7))
        plt.plot(good, bad, [0, 1], [0, 1], 'r')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('Good')
        plt.ylabel('Bad')
        plt.title('Gini = %.4f (%s)' % (g, self.score_name_))
        plt.show()
        if img_path is not None:
            plt.savefig(join(img_path, self.score_name_) + '_Gini.png', format='png')

    def ks_plot(self, img_path=None):
        """绘制 KS 曲线图，并计算 KS 系数。无返回值   \n
        参数:
        ------------
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        ks, good, bad = self.ks_, self.good_curve_, self.bad_curve_
        good_score = good[self.ks_score_]
        bad_score = bad[self.ks_score_]
        plt.figure(figsize=(7, 7))
        plt.plot(good.index, good, 'g', bad.index, bad, 'r')
        plt.plot([self.ks_score_, self.ks_score_], [good_score, bad_score], 'b')
        plt.xlabel('Score')
        plt.ylabel('CumPct')
        plt.title('KS = %.4f (%s)' % (ks, self.score_name_))
        plt.legend(['Good', 'Bad'], loc='lower right')
        plt.ylim([0, 1])
        plt.show()
        if img_path is not None:
            plt.savefig(join(img_path, self.score_name_) + '_KS.png', format='png')

    def giniks_plot(self, img_path=None):
        """绘制 GiniKS 曲线图。它有两个子图，一个子图显示 ROC 曲线，一个子图显示 KS 曲线。  \n
        参数:
        ------------
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        f = plt.figure(figsize=(12, 5.5))

        ax1 = f.add_subplot(1, 2, 1)
        ax1.plot(self.good_curve_, self.bad_curve_, [0, 1], [0, 1], 'r')
        ax1.set_xlabel('Good')
        ax1.set_ylabel('Bad')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Gini: {0:.4f} ({1})'.format(self.gini_, self.score_name_))

        ax2 = f.add_subplot(1, 2, 2)
        ks, good, bad = self.ks_, self.good_curve_, self.bad_curve_
        good_score = good[self.ks_score_]
        bad_score = bad[self.ks_score_]
        ax2.plot(good.index, good, 'g', bad.index, bad, 'r')
        ax2.plot([self.ks_score_, self.ks_score_], [good_score, bad_score], 'b')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('CumPst')
        ax2.set_title('KS = %.4f (%s)' % (ks, self.score_name_))
        ax2.set_ylim([0, 1])
        ax2.legend(['Good', 'Bad'], loc='lower right')

        f.show()
        if img_path is not None:
            plt.savefig(join(img_path, self.score_name_) + '_GiniKS.png', format='png')

    def lift_plot(self, bins=10, gain=False, img_path=None):
        """绘制模型的提升图/增益图, 图中点的 x 坐标值是各 score 各区间的右端点。 \n
        提升图：随着分数cutoff 点的增大，预测为 bad 的样本中的 badrate 相对于基准 badrate 的提升倍数 \n
        增益图：随着分数cutoff 点的增大，预测为 bad 的样本中的 badrate。增益图为提升图的绝对值版本，只是纵轴不同。
        参数:
        ------------
        bins : int, 把分数分为多少个等深区间，默认10个。\n
        gain : bool, 默认为False, 绘制提升图。如果设为True，将绘制增益图。\n
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        score, y, name = self.score_, self.y_, self.score_name_
        df = pd.DataFrame({'score': score, 'y': y})
        badRate = y.mean()
        group = pd.qcut(score, bins)
        bad = df.groupby(group).y.agg(['sum', 'count'])  # 此处 bad 会自动按 score 升序排列
        bad = bad.cumsum()  # lift 是累积正确率，不是各分组内的正确率
        if not gain:
            chart_name = 'Lift'
            bad['lift'] = bad['sum'] / bad['count'] / badRate  # 相对于基准的提升倍数
        else:
            chart_name = 'Gain'
            bad['lift'] = bad['sum'] / bad['count']   # 绝对提升值，即预测为 bad 的样本中的badrate

        # 传统提升图的 x 轴是 depth = 预测为bad的个数/总样本观测数，对于等深区间就是[1/bins, 2/bins, ...]，很好推算
        # 此处用 score 各个分组区间的右端点作为 x 轴，能在 lift 图中展示各点对应的score值。
        barL = [eval(i.replace('(', '[')) for i in bad.index]
        x = [round(i[1], 4) for i in barL]

        plt.figure(figsize=(7, 7))
        plt.plot(x, bad.lift, 'b.-', ms=10)
        plt.xlabel('score')
        plt.ylabel('{1} (base: {0:.2f}%)'.format(100 * badRate, chart_name))
        plt.title('{1} ({0})'.format(name, chart_name))
        if img_path is not None:
            plt.savefig(join(img_path, name) + '_{}.png'.format(chart_name), format='png')

    def divg_plot(self, bins=30, img_path=None):
        """绘制分数的散度图，bad, good分组的直方图均做了归一化。\n
        参数:
        ----------
        bins : int, 把分数分为多少个等宽区间。\n
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片"""
        score, y, name = self.score_, self.y_, self.score_name_
        df = pd.DataFrame(data={'score': score, 'y': y})
        score0 = df.score[df.y == 0]
        score1 = df.score[df.y == 1]
        u0 = score0.mean()
        v0 = score0.var()
        u1 = score1.mean()
        v1 = score1.var()
        div = 2 * (u0 - u1) ** 2 / (v0 + v1)
        plt.figure(figsize=(7, 7), facecolor='w')
        plt.hist(score0, bins=bins, alpha=0.3, color='g', normed=True, label='Good')
        plt.hist(score1, bins=bins, alpha=0.3, color='r', normed=True, label='Bad')
        # kde = sm.nonparametric.KDEUnivariate(score)  # 核密度估计
        # kde.fit()
        #plt.plot(kde.support, kde.density, 'k', lw=2, label='density')
        plt.legend(loc='upper left')
        plt.xlabel('Score')
        plt.ylabel('Freq')
        plt.title('Divergence = %.4f (%s)' % (div, name))
        plt.show()
        if img_path is not None:
            plt.savefig(join(img_path, name) + '_divg.png', format='png')

    def cutoff(self, q=None, step=10):
        """把评分等分为若干bins，并计算 badRate 和 recall 表格.\n
        参数:
        ----------
        q : int, 等深分组的组数. \n
        step : 等宽分组的步长，当 q 为 None 时才使用 step 参数  \n
        返回值:
        ----------
        cutoff_df: dataframe, 计算了以各个点作为cutoff点时，通过的bad，good占比、拒绝的bad, good占比。
        """
        score, y = self.score_, self.y_
        if q is not None:
            bins = pd.qcut(score, q)
        else:
            if step == 0:
                raise Exception("step 步长不能为0")
            bins = score // step * step
        df = pd.crosstab(bins, y)
        df['colName'] = self.score_name_
        df = df.reindex(columns=['colName', 0, 1])
        df['All'] = df[0] + df[1]
        df['binPct'] = df['All'] / df['All'].sum()
        df['badRate'] = df[1] / df['All']
        reject_df = df[[0, 1, 'All']].cumsum() / df[[0, 1, 'All']].sum()
        reject_df['rejBadRate'] = df[1].cumsum() / df['All'].cumsum()
        reject_df = reject_df.shift().fillna(0)
        approve_df = 1 - reject_df[[0, 1, 'All']]
        approve_badrate = {}
        for i in df.index:
            approve_badrate[i] = df[1][i:].sum() / df['All'][i:].sum()
        approve_df['passBadRate'] = pd.Series(approve_badrate)
        reject_df = reject_df.rename(columns={0: 'rejGood', 1: 'rejBad', 'All': 'rejAll'})
        approve_df = approve_df.rename(columns={0: 'passGood', 1: 'passBad', 'All': 'passAll'})
        df = pd.concat([df, approve_df, reject_df], axis=1)
        return df

    def compare_to(self, *thats, img_path=None):
        """与其他 ModelEval 对象比较,把它们的 ROC 曲线画在同一个图上。\n
        参数:
        ----------
        thats : 1~5 个 ModelEval 对象。\n
        img_path : 若提供路径名，将会在此路径下，以score_name为名保存图片\n"""
        that, *args = thats
        if img_path is not None:
            img_path = join(img_path, self.score_name_) + '_'
        gini_compare(self, that, *args, img_path=img_path)

    def __setattr__(self, name, value):
        if name in self.__dict__:  # 一旦赋值，便不可修改、删除属性
            raise AttributeError("can't set readonly attribute {}".format(name))
        self.__dict__[name] = value

    def __delattr__(self, name):
        raise AttributeError("can't del readonly attribute {}".format(name))

    def __str__(self):
        return "ModelEval {0}: Gini={1:.4f}, KS={2:.4f}\n".format(self.score_name_, self.gini_, self.ks_)

    def __repr__(self):
        return self.__str__()


def gini_compare(model_eval1, model_eval2, *args, img_path=None):
    """对比不同的ModelEval对象的性能，把它们的 ROC 曲线画在同一张图上。注意：最多可同时比较 6 个对象。 \n
    参数:
    ----------
    model_eval1: ModelEval 对象。  \n
    model_eval2: ModelEval 对象。  \n
    args: 0 ~ 4 个 ModelEval 对象。  \n
    img_path : 若提供路径名，将会在此路径下，以gini_compare为名保存图片\n"""
    plt.figure(figsize=(7, 7))
    plt.axis([0, 1, 0, 1])
    plt.title('Gini Compare')
    plt.xlabel('Good')
    plt.ylabel('Bad')
    score_list = [model_eval1, model_eval2] + list(args)
    k = len(score_list)
    if k > 6:  # 最多只支持6个模型的比较
        k = 6
        score_list = score_list[:6]
        print("Waring: 最多只支持6个模型的比较,其余模型未比较")
    color = ['b', 'r', 'c', 'm', 'k', 'y'][:k]
    lable = []
    for mod, cor in zip(score_list, color):
        plt.plot(mod.Good_curve, mod.Bad_curve, cor)
        lable.append('{0}: {1:.4f}'.format(mod.Score_name, mod.Gini))
    plt.plot([0, 1], [0, 1], color='grey')
    plt.legend(lable, loc='lower right')
    plt.show()
    if img_path is not None:
        plt.savefig(img_path + '_gini_compare.png', format='png')


def vif(detail_df):
    """计算各个变量的方差膨胀系数 VIF  \n
    参数:
    ----------
    detail_df: dataframe, 明细数据框，一般由清洗好、转换好的各个woe变量组成   \n
    返回值:
    ----------
    vif_sr: series, 各个变量的方差膨胀系数"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    col = dict(zip(detail_df.columns, range(detail_df.shape[1])))
    vif_df = {}
    df_arr = np.array(detail_df)
    for i in col:
        vif_df[i] = variance_inflation_factor(df_arr, col[i])
    vif_sr = pd.Series(vif_df)
    vif_sr.index.name = 'colName'
    vif_sr.name = 'VIF'
    return vif_sr


def date2str(date_obj, sep='-'):
    """把datetime.date对象转换成日期字符串, 转换的格式由 sep 参数决定  \n 
    参数:
    ----------
    date_obj: datetime.date or datetime.datetime 对象.  \n
    sep: 分隔符，指定转换成什么格式的日期字符串:   \n
        '-' 表示转换成 '2017-06-01' 格式的日期  \n
        '/' 表示转换成 '2017/06/01' 格式的日期  \n
        ''  表示转换成 '20170601' 格式的日期
    返回值:
    ----------
    date_str: str, 日期字符串"""
    assert sep in ('-', '/', ''), "仅支持sep 在('-', '/', '')中取值"
    str_f = '%Y{0}%m{0}%d'.format(sep)
    date_str = date_obj.strftime(str_f) if pd.notnull(date_obj) else 'NaT'
    return date_str


def str2date(date_str, sep='-'):
    """把日期字符串转换成 datetime.date对象。  \n
     参数:
     ------------
     date_str: str, 表示日期的字符串。  \n
     sep: 分隔符，说明date_str是什么格式的日期。只要date_str的前 8/10 位表示日期就行，多余的字符串不影响解析； \n
        '-' 表示date_str是'2017-06-01' 格式的日期,   \n
        '/' 表示date_str是'2017/06/01' 格式的日期；  \n
        ''  表示date_str是'20170601' 格式的日期。   \n
    返回值:
    -----------
    date_obj: datetime.date 对象。"""
    if sep in ('-', '/'):
        return date(*[int(i) for i in date_str[:10].split(sep)])
    elif sep == '':
        return date(int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8]))


def time_period(time_sr, freq='week'):
    """根据日期列，生成对应的周/月/年，用来按周/月/年汇总其他数据。   \n
    参数:
    -----------
    time_sr: series,日期字符串('2017-06-01' 或 '2017/06/01' 或 '20170601' 格式，多余字符串不影响） \n
        或者 datetime.date/datetime.datetime 对象   \n
    freq: str, 期望转换成的周期，可选取值有 'week', 'month' or 'year'  \n  
    返回值:
    -----------
    period_sr: series_of_str, 返回对应的周期序列。"""
    from datetime import timedelta
    if freq.upper() == 'WEEK':
        # 字符串格式的日期，先转换成 datetime.date
        if isinstance(time_sr.iloc[0], str):
            date_str = time_sr.iloc[0]
            sep = date_str[4]
            if sep in ('-', '/'):
                time_sr = time_sr.apply(str2date, sep=sep)
            else:
                time_sr = time_sr.apply(str2date, sep='')

        return time_sr.apply(
            lambda x: (x - timedelta(days=x.weekday())).strftime('%Y-%m-%d') if pd.notnull(x) else 'NaT')

    elif freq.upper() == 'MONTH':
        # 字符串格式的日期
        if isinstance(time_sr.iloc[0], str):
            date_str = time_sr.iloc[0]
            sep = date_str[4]
            if sep == '-':
                return time_sr.apply(lambda x: x[:7] )
            elif sep == '/':
                return time_sr.apply(lambda x: x[:7].replace('/', '-'))
            else:  # '20170601' 格式的日期串
                return time_sr.apply(lambda x: x[:4] + '-' + x[4:6])
        else:  # date 或 datetime 对象
            return time_sr.apply(lambda x: x.strftime('%Y-%m'))

    elif freq.upper() == 'YEAR':
        if isinstance(time_sr.iloc[0], str):
            return time_sr.str[:4]
        else:
            return time_sr.apply(lambda x: x.strftime('%Y'))

    else:
        raise Exception("Unkown freq {}: it should be in ('week','month','year')".format(freq))


def dpd_join(detail_df, cols=('fpd10', 'spd10')):
    """合并多个目标变量，得到一个综合的目标变量，比如把 y 定义为前 3 期逾期10+ 。合并逻辑：  \n
    1、任意一个目标变量的某一行不可观测（取值为空）时，综合目标变量的对应行便不可观测。  \n
    2、所有目标变量均可观测时，任何一个目标变量的某一行取1，综合目标变量的对应行就取1。  \n
    参数:
    ----------
    detail_df: dataframe, 包含目标变量的明细数据框  \n
    cols: list, detail_df 中，目标变量的名字  \n
    返回值:
    ----------
    target: series, 综合的目标变量
    """
    target = (detail_df[cols].sum(axis=1) >= 1) * 1

    logic = pd.Series([False]*len(detail_df), index=detail_df.index)
    for col in cols:
        logic = logic | detail_df[col].isnull()
    target.loc[logic] = np.nan
    return target


def fscore(X, y, cols=None):
    """计算数值型单变量的 fscore 指标，适用2分类或多分类。fscore可用来作为特征选择的标准，值越大，表示此变量越有用。\n
    参数:
    ----------
    X: dataframe, 特征变量的数据框, X 的特征都是  \n
    y: series or 1darray, 目标变量，与 X 观测数相等  \n
    cols: list or Index, X 中，特征变量的列名，X 中可以包括额外的列，而 cols 参数指定了需要计算 fscore 的列。  \n
        默认值 None 表示使用 X 中的所有列。  \n
    返回值:
    ----------
    cols_score: series, 各个特征变量的 fscore """

    y = np.array(y)  # 防止 y 的index 与 X 的未对齐
    g = X.groupby(y)
    if cols is None:
        cols = X.columns
    avg = g[cols].mean()
    avg_overall = X[cols].mean()
    var = g[cols].var()
    # 计算各分类中心与全局中心的距离和，与各类内部方差的和的比值
    f_score = ((avg - avg_overall) ** 2).sum() / var.sum()
    return f_score


def train_test_logic(sample, train, test, key=config.PRIM_KEY):
    """返回两个逻辑序列组成的数据框，分别指示 sample 中的观测是否在 train, test 中。 \n
    当从 sample 中随机抽样出 train, test 样本后，需要知道原 sample 中，哪此观测在 train 中，哪些观测在 test 中，\n
    这样当 sample 中的数据变更后，才能够把变更同步到 train, test 数据框中去。 \n
    参数:
    ----------
    sample: dataframe, 随机抽样的源数据  \n
    train: dataframe, 抽样结果之训练样本   \n
    test: dataframe, 抽样结果之测试样本  \n
    key: str, sample 中，标识观测的列名，即逻辑主键名。此列不能有重复。  \n
    返回值:
    ----------
    logic_df: dataframe, 与 sample 等长。包含2个布尔列：  \n
        名为train的列，指示 sample 中的观测是否在 train 中。  \n
        名为text的列，指示 sample 中的观测是否在 test 中。  \n"""
    logic_df = pd.DataFrame()
    logic_df['train'] = sample[key].isin(train[key])
    logic_df['test'] = sample[key].isin(test[key])
    return logic_df


def train_test_update(sample, logic_df):
    """当源数据 sample 作了修改以后，更新 train, test 数据集，同时确保训练样本还是同一批观测（若重新在 sample 上抽样， \n
    则新抽样得到的 train，与原训练样本，其包含的观测集是不同的）。  \n
    参数:
    ----------
    sample: dataframe, 源数据集  \n
    logic_df: dataframe, train_test_logic 函数的返回值 \n
    返回值:
    ----------
    train: dataframe, 同步了 sample 的变更后的训练样本，与旧训练样本包含相同的观测集，因此在此样本上的计算指标均具有  \n
        可比性，可与旧训练样本的相应指标作比较  \n
    test: dataframe, 同步了 sample 的变更后的测试样本，其特点类似 train。"""
    train = sample.loc[logic_df['train']]
    test = sample.loc[logic_df['test']]
    return train, test


def prob2odds(prob_bad):
    """把概率转换成 log(odds)
    参数:
    -----------
    prob_bad: float, series or 1darray, bad 的概率.  \n
    返回值:
    ----------
    log_odds: 类型同 prob_bad, 对应的 log(odds)"""
    return np.log((1 - prob_bad) / prob_bad)


def prob2score(prob_bad, A=427.01, B=57.7078):
    """把概率转换成评分卡的得分  \n
    参数:
    -----------
    prob_bad: float, series or 1darray, bad 的概率.  \n
    A: float, 分数转换的参数A   \n
    B: float, 分数转换的参数B   \n
    返回值:
    ----------
    score: 类型同 prob_bad, 对应的评分卡分数"""
    return A + B * prob2odds(prob_bad)


def offset_transform(prob_model, badrate_actual, badrate_dev):
    """offset 法 adjusting ：进行了 oversampling 抽样的模型，把模型的预测概率转化为实际预测概率  \n
    参数:
    -----------
    prob_model: 1darray or series, 基于 oversampled 的样本训练出来的模型预测的 bad 概率  \n
    badrate_actual: float, 原始样本的 badrate  \n
    badrate_dev: float, oversampled 的用于开发模型的样本 badrate   \n
    返回值:
    ----------
    prob_actual: 1darray or series, 实际预测 bad 概率"""
    good_dev = 1 - badrate_dev
    good_actual = 1 - badrate_actual
    prob_actual = prob_model * good_dev * badrate_actual /  \
                 ((1 - prob_model) * badrate_dev * good_actual + prob_model * good_dev * badrate_actual)
    return prob_actual


def offset_const(const_model, badrate_actual, badrate_dev):
    """offset 法 adjusting ：进行了 oversampling 抽样的模型，把模型的截矩参数转化为实际的截矩参数  \n
    参数:
    -----------
    const_model: float, 基于 oversampled 的样本训练出来的模型的截矩参数  \n
    badrate_actual: float, 原始样本的 badrate  \n
    badrate_dev: float, oversampled 的用于开发模型的样本 badrate   \n
    返回值:
    ----------
    const_actual: float, 真实样本中, 模型的截矩参数。真实样本模型的其他参数，与抽样模型的其他参数相同"""
    good_dev = 1 - badrate_dev
    good_actual = 1 - badrate_actual
    offset = np.log(good_dev * badrate_actual / (badrate_dev * good_actual))
    const_actual = const_model - offset
    return const_actual


def sample_weight(badrate_actual, badrate_dev):
    """sample weight 法 adjusting: 进行了 oversampling 抽样的模型, 计算 bad 和 good 样本的权重  \n
    参数:
    -----------
    badrate_actual: float, 原始样本的 badrate  \n
    badrate_dev: float, oversampled 的用于开发模型的样本 badrate   \n
    返回值:
    ----------
    weights: dict, bad 和 good 样本的权重。"""
    good_dev = 1 - badrate_dev
    good_actual = 1 - badrate_actual
    weights = {}
    weights[0] = good_actual / good_dev
    weights[1] = badrate_actual / badrate_dev
    return weights


def chi_test(x, y):
    """皮尔逊卡方独立检验: 衡量特征的区分度  \n
    参数:
    -----------
    x: array-like, 一维，离散型特征变量  \n
    y: array-like，一维，另一个离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n
    返回值:
    ----------
    chi_result: dict, 卡方检验结果, 其中:  \n
        键'Value'是卡方值,  \n
        键'Prob'是检验的 p 值，值越小， x 与 y 之间的关联性越强/ 区分度越大   \n
        键'DF'是卡方值的自由度.
    """
    tab = pd.crosstab(x, y).fillna(0)
    from scipy.stats import chi2_contingency
    chi_value, p_value, def_free, _ = chi2_contingency(tab)
    return {'DF': def_free, 'Value': chi_value, 'Prob': p_value}


def likelihood_ratio_test(x, y):
    """多项分布的似然比检验： 衡量特征的区分度. 皮尔逊卡方独立检验是似然比检验的近似  \n
    参数:
    -----------
    x: array-like, 一维，离散型特征变量  \n
    y: array-like，一维，二分类离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n
    返回值:
    ----------
    likelihood_result: dict, 卡方检验结果, 其中:  \n
        键'Value'是似然比的值,  \n
        键'Prob'是检验的 p 值，值越小， x 与 y 之间的关联性越强/ 区分度越大   \n
        键'DF'是似然比的自由度."""
    from scipy.stats import chi2_contingency  # 似乎没有找到直接计算似然比的函数, 用此函数计算出期望个数
    tab = pd.crosstab(x, y).fillna(0)
    chi, p, df_free, e_of_tab = chi2_contingency(tab)   # e_of_tab 即列联表中各项的期望个数
    likelihood = (2 * tab * np.log(tab / e_of_tab)).sum().sum()
    p_value = chi2.sf(likelihood, df_free)
    return {'Value': likelihood, 'Prob': p_value, 'DF': df_free}


def odds_ratio(x, y):
    """优势比检验: 仅适用于 x, y 均是二分类随机变量的情形. 优势比的值越远离1, 表示特征 x 的区分度越强  \n
    参数:
    -----------
    x: array-like, 一维，离散型特征变量  \n
    y: array-like，一维，另一个离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n
    返回值:
    ----------
    odds_result: dict, 优势比及其 95% 的置信区间, 置信区间不包含1表明存在显著差异. 其中:  \n
        键'Value'是优势比的值,  \n
        键'left'是优势比的 95% 置信区间的左端点   \n
        键'right'是优势比的 95% 置信区间的右端点."""
    tab = pd.crosstab(x, y).fillna(0)
    n11 = tab.iloc[0, 0]
    n12 = tab.iloc[0, 1]
    n21 = tab.iloc[1, 0]
    n22 = tab.iloc[0, 1]
    if all(n11, n12, n21, n22):
        ratio = n11 * n22 / (n21 * n12)
        var = 1 / n11 + 1 / n12 + 1 / n21 + 1 / n22
    else:
        ratio = (n11 + 0.5) * (n22 + 0.5) / ((n12 + 0.5) * (n21 + 0.5))
        var = 1 / (n11 + 0.5) + 1 / (n12 + 0.5) + 1 / (n21 + 0.5) + 1 / (n22 + 0.5)
    z = 1.96  # 正态分布 97.5% 分位数
    confidence_left = ratio * np.exp(-z * np.sqrt(var))
    confidence_right = ratio * np.exp(z * np.sqrt(var))
    return {'Value': ratio, 'left': confidence_left, 'right': confidence_right}


def f_test(x, y):
    """F检验: 衡量一个连续变量和一个离散变量之间的关联性.  \n
    参数:
    -----------
    x: array-like, 一维，连续型特征变量  \n
    y: array-like，一维，离散型特征变量。当 y 为目标变量时，此检验可以衡量特征区的分度  \n
    返回值:
    ----------
    odds_result: dict, 优势比及其 95% 的置信区间, 置信区间不包含1表明存在显著差异. 其中:  \n
        键'Value'是优势比的值,  \n
        键'left'是优势比的 95% 置信区间的左端点   \n
        键'right'是优势比的 95% 置信区间的右端点."""
    r = len(pd.unique(y))
    n = len(y)
    df = pd.DataFrame({'x': x, 'y': y})
    group = df.groupby('y')
    mean_i = group['x'].mean()
    mean_all = df['x'].mean()
    n_i = group[x].count()
    sstr = (n_i * (mean_i - mean_all)**2).sum()   # 组之间的平方和
    sstr_mean = sstr / (r - 1)

    df['mean_i'] = df['y'].map(mean_i)  # 以 y 的取值为分组, 各个组内的 x 的平均值
    sse = ((df[x] - df['mean_i'])**2).sum()  # 组内的平方和
    sse_mean = sse / (n - r)

    f = sstr_mean / sse_mean




