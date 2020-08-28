import numpy as np
from factorRegression import FactorRegression
import statsmodels.api as sm


def dataType1(x_len, slices, is_list, direction="vertical"):
    """
    x, y都是（x_len天 * slices支股票）的矩阵
    满足关系：
    y = 1 + 2 * x
    alpha = 1
    beta = 2
    weights系数全1
    """

    x_data = np.random.randint(100, size=(x_len, slices)) + 0.01
    y_data = 1 + 2 * x_data + np.random.randn(x_len, slices) / 10
    if is_list:
        weights = np.random.rand(x_len) if direction == "vertical" else np.ones(slices)
    else:
        weights = np.random.rand(x_len, slices)
    print("input:\n", x_data, "\n\n", y_data, "\n\n", weights, "\n\n")
    return x_data, y_data, weights


def dataType2(x_len):
    """
    x是（x_len天 * 2个factors/stocks）的矩阵
    y是（x_len天 * 一列y值）的矩阵
    满足关系：
    y = 1 + 3 * x1 + 2 * x2
    alpha = 1
    beta = [3, 2]
    weights系数全1
    """

    def f(row):
        return 1 + 3 * row[0] + 2 * row[1] + np.random.randn() / 10

    x_data = np.random.randint(100, size=(x_len, 2))
    y_data = np.apply_along_axis(f, 1, x_data).reshape((-1, 1))
    weights = np.random.rand(x_len)
    print("input:\n", x_data, "\n\n", y_data, "\n\n", weights, "\n\n")
    return x_data, y_data, weights


def printf(ls):
    print("\nsolution=", ls.solution)
    print("\nalpha=", ls.alpha)
    print("\nbeta=", ls.beta)
    print("\ndf_tot=", ls.df_tot)
    print("\ndf_reg=", ls.df_reg)
    print("\ndf_err=", ls.df_err)
    print("\nstd_err=", ls.std_err)
    print("\npredicted=", ls.predicted)
    print("\nresids=", ls.resids)
    print("\njarque_bera=", ls.jarque_bera)
    print("\nybar=", ls.ybar)
    print("\nss_tot=", ls.ss_tot)
    print("\nss_reg=", ls.ss_reg)
    print("\nss_err=", ls.ss_err)
    print("\ndurbin_watson=", ls.durbin_watson)
    print("\nrsq=", ls.rsq)
    print("\nrsq_adj=", ls.rsq_adj)
    print("\nms_err=", ls.ms_err)
    print("\nms_reg=", ls.ms_reg)
    print("\nfstat=", ls.fstat)
    print("\nfstat_sig=", ls.fstat_sig)
    print("\n_se_all=", ls._se_all)
    print("\n_tstat_all=", ls._tstat_all)
    print("\n_pvalues_all=", ls._pvalues_all)
    print("\nse_alpha=", ls.se_alpha)
    print("\nse_beta=", ls.se_beta)
    print("\ntstat_alpha=", ls.tstat_alpha)
    print("\ntstat_beta=", ls.tstat_beta)
    print("\npvalue_alpha=", ls.pvalue_alpha)
    print("\npvalue_beta=", ls.pvalue_beta)


def resolveOls(x_data, y_data, direction="vertical", window=None, extend_window=True):
    fr = FactorRegression(x_data=x_data, y_data=y_data, direction=direction)
    ls = fr.get_ols(window=window, extend_window=extend_window)
    printf(ls)


def resolveWls(x_data, y_data, weights, direction="vertical", window=None, extend_window=True):
    fr = FactorRegression(x_data=x_data, y_data=y_data, direction=direction, weights=weights)
    ls = fr.get_wls(window=window, extend_window=extend_window)
    printf(ls)
    
    
def resolveCorr(x_data, y_data, direction="vertical", window=None, extend_window=True):
    fr = FactorRegression(x_data=x_data, y_data=y_data, direction=direction)
    co = fr.get_corr(window=window, extend_window=extend_window)
    print(co.pearson)
    print(co.spearman)


if __name__ == "__main__":
    x1, y1, w1 = dataType1(6, 5, False)
    x2, y2, w2 = dataType2(6)
    
    xx2 = [[1, r[0], r[1]] for r in x2]
    yy2 = [r[0] for r in y2]
    wls = sm.WLS(yy2, xx2, w2).fit()
    print("\nsm.WLS.rsq=", wls.rsquared)
    print("\nsm.WLS.tvalues=", wls.tvalues)
    print("\nsm.WLS.pvalue=", wls.pvalues)

    resolveOls(x1, y1)
    resolveOls(x1, y1, window=4)
    resolveOls(x1, y1, window=4, extend_window=False)
    resolveOls(x1, y1, direction="horizontal")
    resolveOls(x1, y1, direction="horizontal", window=4, extend_window=False)
    resolveOls(x2, y2)
    resolveOls(x2, y2, window=4)

    resolveWls(x1, y1, w1)
    resolveWls(x1, y1, w1, window=4)
    resolveWls(x1, y1, w1, window=4, extend_window=False)
    resolveWls(x1, y1, w1, direction="horizontal")
    resolveWls(x1, y1, w1, direction="horizontal", window=4, extend_window=False)
    resolveWls(x2, y2, w2)
    resolveWls(x2, y2, w2, window=4)
    resolveWls(x2, y2, w2, window=4, extend_window=False)

    x11, y11, w11 = dataType1(6, 5, True)
    resolveWls(x11, y11, w11)
    resolveWls(x11, y11, w11, window=4)
    resolveWls(x11, y11, w11, window=4, extend_window=False)
    x11, y11, w11 = dataType1(6, 5, True, "horizontal")
    resolveWls(x11, y11, w11, direction="horizontal")
    resolveWls(x11, y11, w11, direction="horizontal", window=4)
    resolveWls(x11, y11, w11, direction="horizontal", window=4, extend_window=False)

    resolveCorr(x1, y1)
    resolveCorr(x1, y1, window=4)
    resolveCorr(x1, y1, window=4, extend_window=False)
    resolveCorr(x1, y1, direction="horizontal")
    resolveCorr(x1, y1, direction="horizontal", window=4)
    resolveCorr(x1, y1, direction="horizontal", window=4, extend_window=False)