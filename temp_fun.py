def odds2score(params,woe_df,A=427.01,B=-57.7078):
    """根据逻辑回归的系数，把odds转换成各箱的得分，输出标准评分卡。
    Parameters
    ----------
    params : series
        逻辑回归得到的各变量的系数
    woe_df : df
        各变量的分箱数据表
    Returns
    ----------
    scorecard_df : df, 标准评分卡"""
    const_df = pd.DataFrame({'colName':'const','woe':1})
    woe_df = pd.concat([const_df,woe_df])
    base_score = A + B * params['const']  # 基础分
    Bbetas = (B * params).reset_index().rename(columns={'level_0':'colName',0:'betas'})
    woe_df = woe_df.merge(Bbetas,on='colName')
    woe_df['score'] = woe_df['betas'] * woe_df['woe'] # 各变量的各箱得分
    woe_df.loc[woe_df.colName == 'const','score'] = base_score
    return woe_df
