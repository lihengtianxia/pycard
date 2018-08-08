# -*- coding: utf-8 -*-
"""Created on Mon Mar  5 20:58:25 2018 @author: 左词
针对各模型算法结果的辅助函数，如决策树、kmeans、pca训练的结果"""
from statsmodels.api import Logit, add_constant
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from .core import ModelEval, prob2score
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
from scipy.stats.distributions import chi2

# tree based models, all has feature_importance_ attribute
def export_feature_importance(clf, cols):
    """输出分类器的变量绝对重要性和相对重要性  \n
    参数:
    -----------
    clf: 分类器，一般是 sklean 包中的、训练好了的、有 feature_importances_ 属性  \n
    cols: list, 分类器使用的训练数据中，各个特征变量的名字  \n
    返回值:
    ----------
    fi: dataframe, 变量重要性，按降序排列。包含 2 列：  \n
        abs_impt, 变量的绝对重要性  \n
        rela_impt: 变量的相对重要性   \n"""
    fi = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
    fi = fi.to_frame().rename(columns={0: 'abs_impt'})
    fi['rela_impt'] = fi['abs_impt'] / fi['abs_impt'].max()
    return fi


# 主成分算法相关
def pca_params(clf, cols):
    """pca 拟合结果中，各主成分的转换参数 beta, 用于从标准化之后的数据集中得到主成分。  \n
    若 X 为标准化之后的数据集，则各主成分的转换公式为 pca_i =  X * beta_i  \n
    参数:
    -----------
    clf: sklearn.decomposition.PCA 对象, 主成分训练器，且已在标准化之后的数据集上训练好了参数  \n
    cols: list, 训练数据中，各列特征的名字  \n
    返回值:
    ----------
    beta: dataframe, n_features * n_components, 各个主成分的转换参数。"""
    col_str = 'pca{i}_{ratio}'  # i 表示第几主成分，ratio 表示此主成分解释的方差比例。
    exp_ratio = [col_str.format(i=str(k + 1), ratio=str(round(r * 100))[:-2]) for k, r in
                 enumerate(clf.explained_variance_ratio_)]
    return pd.DataFrame(clf.components_.transpose(), cols, exp_ratio)


def pca_param_raw(clf, cols, train_source):
    """pca 拟合结果中，各主成分的转换参数 beta，用于从非标准化的数据集中得到主成分。
    若 X 为未标准化的数据集，则各主成分的转换公式为：pca_i =  (X - X.mean()) * beta_i  \n
    参数:
    -----------
    clf: sklearn.decomposition.PCA 对象, 主成分训练器，且已在标准化之后的数据集上训练好了参数  \n
    cols: list, 训练数据中，各列特征的名字  \n
    train_source: 原始未标准化的数据集，训练数据集由 train_source 经过标准化得到  \n
    返回值:
    ----------
    beta: dataframe, n_features * n_components, 各个主成分的转换参数。"""
    beta = pca_params(clf, cols)
    beta = beta.div(train_source[cols].std(), axis=0)
    beta['mean'] = train_source[cols].mean()
    return beta


def pca_explain_ratio(clf, img_path=None):
    """绘制各个主成分解释的方差比例图，即跌崖碎石图  \n
    参数:
    ------------
    clf: sklearn.decomposition.PCA 对象, 主成分训练器，且已在标准化之后的数据集上训练好了参数  \n
    img_path : 若提供路径名，将会在此路径下，以variance_ratio为名保存图片"""
    pca_ratio = clf.explained_variance_ratio_
    pca_ratio_cum = pca_ratio.cumsum()
    pcas = range(1, len(pca_ratio)+1)
    fig = plt.figure(figsize=(7, 7))
    ax1 = fig.add_subplot(111)
    ax1.plot(pcas, pca_ratio, 'b-v', label='ratio')
    ax1.set_ylabel('variance ratio')
    ax1.legend(loc='center left')

    ax2 = ax1.twinx()
    ax2.plot(pcas, pca_ratio_cum, 'r-o', label='cum_ratio')
    ax2.set_ylabel('cumulate variance ratio')
    ax2.legend(loc='center right')

    ax1.set_xlabel('PCA order')
    plt.show()

    if img_path is not None:
        fig.savefig(join(img_path, 'variance_ratio.png'), format='png')


# statsmodel.api.Logit 相关函数
def logit_variales(logit_result):
    """从逻辑回归结果中，提取所有自变量  \n
    参数:
    -----------
    logit_result: statsmodel.api.Logit.fit() 返回结果对象。"""
    return logit_result.params.index.drop('const')


def logit_stats(logit_result, n, p):
    """计算模型结果的各个统计指标.   \n
    参数:
    ------------
    logit_result: statsmodel.api.Logit.fit() 返回结果对象   \n
    n: int, 样本的个数  \n
    p: int, 特征的个数, 包括常数项  \n

    返回值:
    -------------
    stats: series, 其中各个元素值说明如下：  \n
        AIC: Akaike information criterion.  `-2*(llf - p)` , p 包括常数项  \n
        BIC: Bayesian information criterion. `-2*llf + ln(nobs)*p` p 包括常数项  \n
        SC: Schwarz criterion. `-LogL + p*(log(nobs))`   \n
        2logl: Value of the -2 * loglikelihood, as (LogL)  \n
    """
    aic = logit_result.aic
    bic = logit_result.bic
    logl = -2 * logit_result.llf  # 逻辑回归的离差
    sc = 2 * (-logit_result.llf + p * (np.log(n)))
    stats = pd.Series([aic, bic, logl, sc], index=["AIC", "BIC", "-2Logl", "SC"])
    return stats


def logit_fit(x_data, y, name='train'):
    """拟合逻辑回归，并绘制 gini,ks 曲线  \n
    参数:
    ----------
    x_data: dataframe, 已清洗好的训练数据的特征变量，函数会自动补上常数项  \n
    y: series or 1darray, 目标变量   \n
    name: 训练模型的名字  \n
    返回值:
    ----------
    result: statsmodel.api.Logit.fit() 返回结果对象  \n
    model_eval: ModelEval, 模型评估对象"""
    model_data = add_constant(x_data)
    logit_reg = Logit(y, model_data)
    result = logit_reg.fit(disp=False)

    prob = result.predict(model_data)
    model_eval = ModelEval(-prob, y, name, plot=False)

    a = "************************************"
    print(a + "  " + name + "  " + a)
    print(result.summary2())
    model_eval.giniks_plot()
    return result, model_eval


def logit_eval(logit_result, test, y, name='train'):
    """根据逻辑回归结果，绘制给定数据的gini,ks曲线。一般用来评估验证、测试数据集\n
    参数:
    ----------
    logit_result: statsmodel.api.Logit.fit() 返回结果对象  \n
    test: dataframe, 测试/验证数据集。函数会自动补上常数项  \n
    y: series 目标变量 \n
    name: 数据集的名字  \n
    返回值:
    ----------
    model_eval: ModelEval, 模型评估对象"""
    col = logit_variales(logit_result)
    model_data = add_constant(test[col])
    prob = logit_result.predict(model_data)
    model_eval = ModelEval(-prob, y, name)
    return model_eval


def logit_predict(logit_result, sample, version='', bins=20):
    """根据逻辑回归结果，给 sample 打上 prob, score, score_group \n
    参数:
    ----------
    logit_result: statsmodel.api.Logit.fit() 返回结果对象  \n
    sample: 明细数据表，应包含所有的入模变量，函数会自动补上常数项  \n
    version: str, 模型的版本。版本号将以后缀添加在 prob, score, group 变量名后面  \n
    bins: int, 指定对分数等深分成几组，分组结果便是 group 列\n
    返回值:
    ----------
    没有返回值，原地修改sample，增加 prob_bad, score, group 三列"""
    col = logit_variales(logit_result)
    x_data = add_constant(sample[col])
    prob, score, group = [i + '_' + version for i in ('prob', 'score', 'group')]
    sample[prob] = logit_result.predict(x_data)
    sample[score] = prob2score(sample[prob])
    sample[group] = pd.qcut(sample[score], bins)


# LogisticRegression 模型结果相关的函数
def logistic_coef(clf, cols):
    """返回逻辑回归的模型参数  \n
    参数:
    ----------
    clf: LogisticRegression 对象，训练好的模型对象  \n
    cols: list or Index, 所用的各个特征的名字, 不用把常数名包含进来  \n
    返回值:
    ----------
    beta: series, 各个特征的模型参数  \n
    """
    const = pd.Series([clf.intercept_], index=['const'])
    betas = pd.Series(clf.coef_, index=cols)
    return pd.concat([const, betas])


def logistic_eval(clf, test, y, name='train'):
    """根据逻辑回归结果，绘制给定数据的gini,ks曲线。一般用来评估验证、测试数据集\n
    参数:
    ----------
    clf: LogisticRegression 对象，训练好的模型对象  \n
    test: dataframe, 测试/验证数据集。函数会自动补上常数项  \n
    y: series or 1darray, 目标变量 \n
    name: 数据集的名字  \n
    返回值:
    ----------
    model_eval: ModelEval, 模型评估对象"""
    prob = clf.predict_prob(test)
    prob_bad = prob[:, 1]
    model_eval = ModelEval(-prob_bad, y, name)
    return model_eval


def logistic_predict(clf, sample, cols, version='', bins=20):
    """根据逻辑回归结果，给 sample 打上 prob, score, score_group \n
    参数:
    ----------
    clf: LogisticRegression 对象，训练好的模型对象  \n
    sample: 明细数据表，应包含所有的入模变量，函数会自动补上常数项  \n
    cols: list or Index, 所用的各个特征的名字, 不用把常数名包含进来  \n
    version: str, 模型的版本。版本号将以后缀添加在 prob, score, group 变量名后面  \n
    bins: int, 指定对分数等深分成几组，分组结果便是 group 列\n
    返回值:
    ----------
    没有返回值，原地修改sample，增加 prob_bad, score, group 三列"""
    x_data = sample[cols]
    prob, score, group = [i + '_' + version for i in ('prob', 'score', 'group')]
    prob = clf.predict_prob(x_data)   # prob 是 n_samples * 2 的数组，第1列是目标变量为 0 的概率
    sample[prob] = prob[:, 1]
    sample[score] = prob2score(sample[prob])
    sample[group] = pd.qcut(sample[score], bins)


class StepwiseLogit:
    """
    Implementation of Stepwise Logistic Regression Model ，sklearn 没有它的实现  \n

    参数:
    -------------
    entry : float (default=0.05),
        forward step's confidence level
    stay : float (default=0.05)
        backward step's confidence level
    """

    def __init__(self, entry=0.05, stay=0.05):
        self.entry = entry  # 默认进入的p值
        self.stay = stay  # 默认保留的p值

    @staticmethod
    def wald_test(result):
        """逐步回归backward的wald检验。result.wald_test_terms也实现了此算法 \n
        参数:
        ----------
        result: statsmodel.api.Logit.fit() 返回结果对象  \n
        返回值:
        ----------
        test_df: dataframe, wald 检验的结果，包含2列：wald_chi2，pvalue_chi2 """
        wald_chi2 = (result.params / result.bse) ** 2  # backward 的 wald 检验统计量，服从自由度为1的卡方分布
        wald_chi2.name = 'wald_chi2'
        pvalue_chi2 = pd.Series(chi2.sf(wald_chi2, 1),
                                index=wald_chi2.index, name='P>chi2')  # backward 的 wald 检验 p 值
        test = pd.concat([wald_chi2, pvalue_chi2], axis=1)
        return test

    def fit(self, X, y, print_detail=False):
        """Stepwise logistic regression. Use Score test for entry, Wald test for remove.
        参数:
        ----------
        X: array-like, n_sample * p_features. 特征变量数据集，程序会自动添加常数项
        y: array-like, 目标变量
		print_detail: bool, 是否打印出逐步回归选择变量的细节
		返回值:
		-----------
		result: 类型同 statsmodels.api.Logit 对象 fit 方法的返回值, 逐步回归选出的模型。"""

        def score_test(Xtest, y_true, y_predict):
            """对step forward进入的变量进行Score检验。函数假设新进入的变量放在最后.
            Xtest包括vars_old(似合模型并给出预测值y_predict的),和var_new（一个待检验的新变量）。
            Score检验假设待检验变量的系数为0，所以Xtest虽然包括了它的数据，但拟合参数是按没有此变量计算出来的。"""
            u = np.dot(Xtest.T, y_true - y_predict)  # 一阶导数
            h = np.dot(Xtest.T * (y_predict * (1 - y_predict)).values.reshape(len(y_predict)), Xtest)  # 二阶导数
            score = np.dot(np.dot(u.T, np.linalg.inv(h)), u)  # score 是 1*1 数组
            p_value = chi2.sf(score, 1)  # Score统计量服从自由度为1的卡方分布
            return score, p_value

        def print_wrap(*obj):
            if print_detail:
                print(*obj)

        X = add_constant(X)
        xenter = ['const']
        xwait = list(X.columns.drop('const'))
        logit_mod = Logit(y, X[xenter])
        logit_res = logit_mod.fit(disp=0)
        y_predict = logit_res.predict(X[xenter])
        step = 0
        while xwait:  # 停止条件1：所有变量都进入了模型
            # entry test
            score = pd.Series(name='Score')
            pvalue = pd.Series(name='P>chi2')
            for xname in xwait:
                tmpX = X[xenter + [xname]]
                score[xname], pvalue[xname] = score_test(tmpX, y, y_predict)

            step += 1
            print_wrap("step {}: Variables Entry test:\n".format(step),
                       pd.concat([score, pvalue], axis=1))  # 打印运行信息

            if pvalue.min() <= self.entry:  # 最显著的变量选进来
                xin = pvalue.argmin()
                xenter.append(xin)
                xwait.remove(xin)
                print_wrap("step {0}: {1} entered.\n".format(step, xin))
            else:  # 停止条件2：没有变量符合进入标准
                print_wrap("Stopped 2: No vars can get entered any more.\n")
                break

            # remove test
            while True:  # 程序运行到这里，说明新增了变量进来
                logit_mod = Logit(y, X[xenter])
                logit_res = logit_mod.fit(disp=0)
                y_predict = logit_res.predict(X[xenter])
                test = logit_res.wald_test_terms().dframe  # wald 检验
                pvalue = test['P>chi2'].iloc[1:]  # 常数项不参与检验

                step += 1
                print_wrap("step {}: Variables remove test:\n".format(step), test)

                if pvalue.max() < self.stay:
                    xout = None
                    print_wrap("step {}: No Variables removed:\n".format(step))
                    break  # 所有变量都是显著的，不剔除变量
                else:
                    xout = pvalue.argmax()
                    xenter.remove(xout)
                    xwait.append(xout)
                    print_wrap("step {0}: {1} removed.\n".format(step, xout))

            # 停止条件3：如果刚进入的变量又剔除
            if xin == xout:
                print_wrap("Stopped 3: last var entered also got removed.\n")
                break
        else:
            print_wrap("Stopped 1: all var available got entered.\n")
        return Logit(y, X[xenter]).fit(disp=0)


# %% kmeans 相关的函数
def kmeans_radium(x, centroids, labels):
    """计算各个类的半径  \n
    参数:
    ----------
    x: dataframe or ndarray, n_samples * n_features, 明细数据表  \n
    centroids: k*p 二维数组, 训练出来的 k 个类的中心   \n
    labels: 1darray, 算法预测的 x 中各个观测所属的类。  \n
    返回值:
    ----------
    raduim: 1darray, k 个类的半径  """
    radium = np.zeros(len(centroids))
    for k in np.unique(labels):
        xk = x[labels == k]
        centroid = centroids[k]
        dist = ((xk - centroid) ** 2).sum(axis=1)
        radium[k] = dist.max()
    return np.sqrt(radium)


def kmeans_avg_dist_inner(x, labels):
    """计算每一类的类内平均距离  \n
    参数:
    ----------
    x: dataframe or ndarray, n_samples * n_features, 明细数据表  \n
    labels: 1darray, 算法预测的 x 中各个观测所属的类。  \n
    返回值:
    ----------
    avg_dist: 1darray, k 个类的类内平均距离"""

    def avg_dist_k(sub_sample):
        """类内平均距离。此函数非常耗时，要计算两两样本点间的距离，时间复杂度O(n*n)"""
        n = len(sub_sample)
        dist = 0
        for i in range(n - 1):
            xi = sub_sample[i:i + 1]
            xj = sub_sample[i + 1:]
            dist_ij = ((xj - xi) ** 2).sum(axis=1)
            dist_ij = np.sqrt(dist_ij)  # 平方根距离
            dist += dist_ij.sum()
        return 2 * dist / (n * (n - 1))

    ks = np.unique(labels)
    avg = np.zeros(len(ks))
    for k in ks:
        xk = x[labels == k]
        avg[k] = avg_dist_k(xk)
    return avg


def kmeans_dbi_value(avg_c, centroids):
    """计算聚类的dbi指标, dbi 是评价聚类质量好坏的指标，k相同时，dbi 越小表示聚类越好。  \n
    参数:
    ----------
    avg_c: 1darray, 各个类的类内平均距离  \n
    centroids: k*p 二维数组, 训练出来的 k 个类的中心   \n
    返回值:
    ----------
    dbi: float, 聚类的dbi指标值"""
    k = len(avg_c)
    avg_ij = [avg_c[i] + avg_c[j] for i in range(k) for j in range(k) if i != j]
    avg_ij = np.array(avg_ij).reshape(k, k - 1)

    dist = lambda i, j: np.sqrt(((centroids[i] - centroids[j]) ** 2).sum())
    dist_uij = [dist(i, j) for i in range(k) for j in range(k) if i != j]
    dist_uij = np.array(dist_uij).reshape(k, k - 1)

    dbi = (avg_ij / dist_uij).max(axis=1)
    return dbi.mean()


def kmeans_avg_dist_with_u(x, centroids, labels):
    """计算各类内的平均质心距离.  \n
    参数:
    ----------
    x: dataframe or ndarray, n_samples * n_features, 明细数据表  \n
    centroids: k*p 二维数组, 训练出来的 k 个类的中心   \n
    labels: 1darray, 算法预测的 x 中各个观测所属的类。  \n
    返回值:
    ----------
    dist_u: 1darray, k 个类的平均质心距离  """
    dist_u = np.zeros(len(centroids))
    for k in np.unique(labels):
        xk = x[labels == k]
        centroid = centroids[k]
        dist = ((xk - centroid) ** 2).sum(axis=1)
        dist_u[k] = np.sqrt(dist).mean()
    return dist_u


def kmeans_weighed_avg(radiums, labels):
    """计算整个聚类的加权平均半径/平均质心距离。函数本身只做加权平均的事情。\n
    参数:
    ----------
    radiums ：1darray, 各个类的半径，或各个类内的平均质心距离  \n
    labels ：1darray, 每条观测属于哪个类的标签，标签从0开始，到n_samples - 1  \n
    返回值:
    ----------
    avg: 用各个类的样本量进行加权的平均半径/平均质心距离
    """
    ni = pd.value_counts(labels).sort_index()
    avg = (radiums * ni.values).sum() / ni.sum()
    return avg
