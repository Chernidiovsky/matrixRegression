import pandas as pd
import numpy as np
from functools import lru_cache
import scipy.stats as scs
from statsmodels.tools import add_constant


def _swap_add_constant(a):
    if a.ndim < 3:
        return add_constant(a, has_constant="raise")
    else:
        a = a.swapaxes(2, 0)  # x date stock
        shape = [1] + list(a.shape)[1:]
        ones = np.ones(shape=shape)
        a = np.vstack((ones, a))  # 1,x date stock
        a = a.swapaxes(2, 0)  # stock date 1,x
        return a


def _rolling_windows(a, window):
    """Creates rolling-window 'blocks' of length `window` from `a`.
    Note that the orientation of rows/columns follows that of pandas.
    Example
    -------
    import numpy as np
    onedim = np.arange(20)
    twodim = onedim.reshape((5,4))
    print(twodim)
    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]
     [12 13 14 15]
     [16 17 18 19]]
    print(rwindows(onedim, 3)[:5])
    [[0 1 2]
     [1 2 3]
     [2 3 4]
     [3 4 5]
     [4 5 6]]
    print(rwindows(twodim, 3)[:5])
    [[[ 0  1  2  3]
      [ 4  5  6  7]
      [ 8  9 10 11]]
     [[ 4  5  6  7]
      [ 8  9 10 11]
      [12 13 14 15]]
     [[ 8  9 10 11]
      [12 13 14 15]
      [16 17 18 19]]]
    """

    if window > a.shape[0]:
        raise ValueError(
            "Specified `window` length of {0} exceeds length of"
            " `a`, {1}.".format(window, a.shape[0])
        )
    if isinstance(a, (pd.Series, pd.DataFrame)):
        a = a.values
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    windows = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    if windows.ndim == 1:
        windows = np.atleast_2d(windows)
    return windows


def _swap_rolling_windows(a, window):
    if a.ndim == 2:
        a = a.transpose()  # date stock
        a = _rolling_windows(a, window)  # date1 date stock
        a = a.transpose((2, 0, 1))  # stock date1 date
        return a
    elif a.ndim == 3:
        a = a.transpose((1, 2, 0))  # date data stock
        a = _rolling_windows(a, window)  # date1 date data stock
        a = a.transpose((3, 0, 1, 2))  # stock date1 date data
        return a
    raise Exception("用于滑窗的矩阵必须是2维或3维")


def _ols(x, y):
    """
    系数矩阵 = (Xt * X)^(-1) * Xt * Y
    """
    if x.ndim < 3 or y.ndim < 3:
        raise np.AxisError("计算ols系数的x, y维度必须都>=3")
    i = x.ndim
    xt = x.swapaxes(i - 2, i - 1)  # swap last 2 axes
    xtx = np.matmul(xt, x)
    xtxi = np.linalg.inv(xtx)
    xty = np.matmul(xt, y)
    result = np.matmul(xtxi, xty)
    return result


def _wls(x, y, w):
    """
    系数矩阵 = (Xt * W * X)^(-1) * Xt * W * Y
    """
    if x.ndim < 3 or y.ndim < 3 or w.ndim < 3:
        raise np.AxisError("计算wls系数的x, y, w 维度必须都>=3")
    xt = x.swapaxes(-2, -1)  # swap last 2 axes
    xtw = np.matmul(xt, w)
    xtwx = np.matmul(xtw, x)
    xtwxi = np.linalg.inv(xtwx)
    xtwy = np.matmul(xtw, y)
    result = np.matmul(xtwxi, xtwy)
    return result


def _confirm_constant(a):
    """Confirm `a` has volumn vector of 1s."""
    a = np.asanyarray(a)
    return np.isclose(a, 1.0).all(axis=0).any()


def _check_constant_params(a, has_const=False, use_const=True, rtol=1e-05, atol=1e-08):
    """Helper func to interaction between has_const and use_const params.
    has_const   use_const   outcome
    ---------   ---------   -------
    True        True        Confirm that a has constant; return a
    False       False       Confirm that a doesn't have constant; return a
    False       True        Confirm that a doesn't have constant; add constant
    True        False       ValueError
    """

    if all((has_const, use_const)):
        if not _confirm_constant(a):
            raise ValueError("Data does not contain a constant; specify has_const=False")
        k = a.shape[-1] - 1
    elif not any((has_const, use_const)):
        if _confirm_constant(a):
            raise ValueError("Data already contains a constant; specify has_const=True")
        k = a.shape[-1]
    elif not has_const and use_const:
        # Also run a quick check to confirm that `a` is *not* ~N(0,1).
        #     In this case, constant should be zero. (exclude it entirely)
        c1 = np.allclose(a.mean(axis=0), b=0.0, rtol=rtol, atol=atol)
        c2 = np.allclose(a.std(axis=0), b=1.0, rtol=rtol, atol=atol)
        if c1 and c2:
            raise ValueError("Data appears to be ~N(0,1). Specify use_constant=False.")
        # `has_constant` does checking on its own and raises VE if True
        try:
            a = _swap_add_constant(a)
        except ValueError as e:
            raise ValueError("X data already contains a constant; please specify has_const=True") from e
        k = a.shape[-1] - 1
    else:
        raise ValueError("`use_const` == False implies has_const is False.")

    return k, a


def _handle_ab(solution, use_const=True):
    solution = np.squeeze(solution)
    if solution.ndim == 1:
        b = solution[1:] if use_const else solution
        b = b.item() if b.size == 1 else b
        a = solution[0] if use_const else None
        return a, b
    elif solution.ndim == 2:
        b = solution[:, 1:] if use_const else solution
        a = solution[:, 0] if use_const else None
        return a, b
    else:
        b = solution[:, :, 1:] if use_const else solution
        a = solution[:, :, 0] if use_const else None
        return a, b


def _clean_xy(x, y, w=None, has_const=False, use_const=True):
    x = np.asanyarray(x) if x is not None else None
    w = np.asanyarray(w) if w is not None else None
    y = np.asanyarray(y)
    k, x = _check_constant_params(x, has_const=has_const, use_const=use_const)
    return x, y, w, k


def _extend_xy(x, y, window, w=None):
    window -= 1
    if x.ndim == 2:
        x_add = np.ones((x.shape[0], window)) * np.nan
        x = np.hstack((x_add, x))
        y_add = np.ones((y.shape[0], window)) * np.nan
        y = np.hstack((y_add, y))
    elif x.ndim == 3:
        x = x.swapaxes(0, 1)
        x_add = np.ones((window, x.shape[1], x.shape[2])) * np.nan
        x = np.vstack((x_add, x))
        x = x.swapaxes(0, 1)
        y = y.swapaxes(0, 1)
        y_add = np.ones((window, y.shape[1], y.shape[2])) * np.nan
        y = np.vstack((y_add, y))
        y = y.swapaxes(0, 1)
    else:
        raise Exception("用于滑窗的矩阵必须是2维或3维")
    if w is None:
        return x, y
    else:
        w_add = np.ones((w.shape[0], window))
        w = np.hstack((w_add, w))
        return x, y, w


def _pinv_extended(x, rcond=1e-15):
    x = np.asarray(x)
    x = x.conjugate()
    u, s, vt = np.linalg.svd(x, False)
    s = np.apply_along_axis(lambda r: [1 / i if i > rcond else 0 for i in r], -1, s)
    res = np.matmul(vt.swapaxes(-1, -2), np.multiply(np.expand_dims(s, axis=-1), u.swapaxes(-1, -2)))
    return res


def _rank_array(a):
    """
    [9 4 2 7] --> [4 2 1 3]
    """
    a = np.array(a)
    b = a.argsort()
    c = np.empty_like(b)
    c[b] = np.arange(1, len(a) + 1)
    return c


class LeastSquares:
    def __init__(self, x, y, w=None, window=None, extend_window=True, has_const=False, use_const=True):
        self.x, self.y, self.w, self.k = _clean_xy(x, y, w, has_const, use_const)
        # print("clean:\n", self.x, "\n\n", self.y, "\n\n", self.w, "\n\n")
        self.extend_window = extend_window
        self.has_const = has_const
        self.use_const = use_const
        self.window = self.n = window

        if self.w is None:
            if self.window is None:
                self.n = self.x.shape[1]
            else:
                if self.extend_window:
                    self.x, self.y = _extend_xy(self.x, self.y, self.window)
                self.x = _swap_rolling_windows(self.x, window=self.window)
                self.y = _swap_rolling_windows(self.y, window=self.window)
            self.solution = _ols(self.x, self.y)
        else:
            if self.window is None:
                self.n = self.x.shape[1]
                eye = np.eye(self.w.shape[1])
            else:
                if self.extend_window:
                    self.x, self.y, self.w = _extend_xy(self.x, self.y, self.window, self.w)
                eye = np.eye(self.window)
                self.x = _swap_rolling_windows(self.x, window=self.window)
                self.y = _swap_rolling_windows(self.y, window=self.window)
                self.w = _swap_rolling_windows(self.w, window=self.window)
            self.w_diag = np.apply_along_axis(lambda r: np.multiply(r, eye), -1, self.w)  # 沿最底层维度转成2维对角阵
            self.solution = _wls(self.x, self.y, self.w_diag)
    
    @property
    def _alpha(self):
        """The intercept term (alpha).
        Technically defined as the coefficient to a column vector of ones.
        """
        return _handle_ab(self.solution, self.use_const)[0]

    @property
    def _beta(self):
        """The parameters (coefficients), excl. the intercept."""
        return _handle_ab(self.solution, self.use_const)[1]

    @property
    def _df_tot(self):
        """Total degrees of freedom, n - 1."""
        return self.n - 1

    @property
    def _df_reg(self):
        """Model degrees of freedom. Equal to k."""
        return self.k

    @property
    def _df_err(self):
        """Residual degrees of freedom. n - k - 1."""
        return self.n - self.k - 1

    @property
    @lru_cache(maxsize=None)
    def _predicted(self):
        """The predicted values of y ('yhat')."""
        return np.matmul(self.x, self.solution)

    @property
    @lru_cache(maxsize=None)
    def _resids(self):
        result = self.y - self._predicted
        if self.w is None:
            return result
        else:
            return np.matmul(np.sqrt(self.w_diag), result)
    
    @property
    def _std_err(self):
        """Standard error of the estimate (SEE).  A scalar.
        For standard errors of parameters, see _se_all, se_alpha, and se_beta.
        """
        return np.sqrt(np.sum(np.square(self._resids), axis=-2) / self._df_err)
    
    @property
    def _jarque_bera(self):
        return np.apply_along_axis(scs.jarque_bera, -2, self._resids)[:, 0]

    @property
    @lru_cache(maxsize=None)
    def _ybar(self):
        """The mean of y."""
        return self.y.mean(axis=-2)

    @property
    @lru_cache(maxsize=None)
    def _ss_tot(self):
        """Total sum of squares."""
        squares = np.square(self.y - np.expand_dims(self._ybar, axis=-2))
        if self.w is None:
            return np.sum(squares, axis=-2)
        else:
            return np.sum(np.matmul(self.w_diag, squares), axis=-2)

    @property
    @lru_cache(maxsize=None)
    def _ss_reg(self):
        """Sum of squares of the regression."""
        squares = np.square(self._predicted - np.expand_dims(self._ybar, axis=-2))
        if self.w is None:
            return np.sum(squares, axis=-2)
        else:
            return np.sum(np.matmul(self.w_diag, squares), axis=-2)

    @property
    @lru_cache(maxsize=None)
    def _ss_err(self):
        """Sum of squares of the residuals (error sum of squares)."""
        return np.sum(np.square(self._resids), axis=-2)

    @property
    def _durbin_watson(self):
        return np.sum(np.square(np.diff(self._resids, axis=-2)) / np.expand_dims(self._ss_err, axis=-1), axis=-2)

    @property
    def _rsq(self):
        """The coefficient of determination, R-squared. rsq = 1 - (ss_err / ss_tot) = ss_reg / ss_tot"""
        return self._ss_reg / self._ss_tot

    @property
    def _rsq_adj(self):
        """Adjusted R-squared."""
        n = self.n
        k = self.k
        return 1.0 - ((1.0 - self._rsq) * (n - 1.0) / (n - k - 1.0))

    @property
    def _ms_err(self):
        """Mean squared error the errors (residuals)."""
        return self._ss_err / self._df_err

    @property
    def _ms_reg(self):
        """Mean squared error the regression (model)."""
        return self._ss_reg / self._df_reg

    @property
    def _fstat(self):
        """F-statistic of the fully specified model."""
        return self._ms_reg / self._ms_err

    @property
    def _fstat_sig(self):
        """p-value of the F-statistic."""
        return 1.0 - scs.f.cdf(self._fstat, self._df_reg, self._df_err)

    # -----------------------------------------------------------------
    @property
    @lru_cache(maxsize=None)
    def _se_all(self):
        """Standard errors (SE) for all parameters, including the intercept."""
        if self.w is None:
            xt = self.x.swapaxes(-2, -1)
            xtx = np.matmul(xt, self.x)
            xtxi = np.linalg.inv(xtx)
            diag = np.diagonal(xtxi, axis1=-2, axis2=-1)
            diag = np.expand_dims(diag, axis=-1)
            err = np.expand_dims(self._ms_err, axis=-1)
            result = np.squeeze(np.matmul(diag, err))
            result = np.sqrt(result)
        else:
            w, x, wresid = self.w_diag, self.x, self._resids
            wx = np.matmul(np.sqrt(w), x)
            try:
                pinv_wx = _pinv_extended(wx)
            except:
                wx = wx[:, self.window - 1:, :, :]
                pinv_wx = _pinv_extended(wx)
                pinv_wx = pinv_wx.swapaxes(0, 1)
                add = np.ones((self.window - 1, pinv_wx.shape[1], pinv_wx.shape[2], pinv_wx.shape[3])) * np.nan
                pinv_wx = np.vstack((add, pinv_wx))
                pinv_wx = pinv_wx.swapaxes(0, 1)
            norm = np.matmul(pinv_wx, pinv_wx.swapaxes(-1, -2))
            nod = wx.shape[-2]
            rank = np.linalg.matrix_rank(x)
            rank = rank[-1][-1] if x.ndim == 4 else rank[-1]
            df_resid = float(nod - rank)
            scale = np.matmul(wresid.swapaxes(-1, -2), wresid) / df_resid
            cov = norm * scale
            result = np.sqrt(np.diagonal(cov, axis1=-2, axis2=-1))
        return result

    @property
    def _se_alpha(self):
        """Standard errors (SE) of the intercept (alpha) only."""
        return _handle_ab(self._se_all, self.use_const)[0]

    @property
    def _se_beta(self):
        """Standard errors (SE) of the parameters, excluding the intercept."""
        return _handle_ab(self._se_all, self.use_const)[1]

    @property
    @lru_cache(maxsize=None)
    def _tstat_all(self):
        """The t-statistics of all parameters, incl. the intecept."""
        return np.squeeze(self.solution) / self._se_all

    @property
    def _tstat_alpha(self):
        """The t-statistic of the intercept (alpha)."""
        return _handle_ab(self._tstat_all, self.use_const)[0]

    @property
    def _tstat_beta(self):
        """The t-statistics of the parameters, excl. the intecept."""
        return _handle_ab(self._tstat_all, self.use_const)[1]

    @property
    @lru_cache(maxsize=None)
    def _pvalues_all(self):
        """Two-tailed p values for t-stats of all parameters."""
        return 2.0 * (1.0 - scs.t.cdf(np.abs(self._tstat_all), self._df_err))

    @property
    def _pvalue_alpha(self):
        """Two-tailed p values for t-stats of the intercept only."""
        return _handle_ab(self._pvalues_all, self.use_const)[0]

    @property
    def _pvalue_beta(self):
        """Two-tailed p values for t-stats of parameters, excl. intercept."""
        return _handle_ab(self._pvalues_all, self.use_const)[1]

    @property
    def _condition_number(self):
        """Condition number of x; ratio of largest to smallest eigenvalue."""
        ev = np.linalg.eig(np.matmul(self.x.swapaxes(-2, -1), self.x))[0]
        return np.sqrt(ev.max(axis=1) / ev.min(axis=1))

    # -----------------------------------------------------------------
    # "Public" results
    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def df_tot(self):
        """Total degrees of freedom, n - 1."""
        return self._df_tot

    @property
    def df_reg(self):
        """Model degrees of freedom. Equal to k."""
        return self._df_reg

    @property
    def df_err(self):
        """Residual degrees of freedom. n - k - 1."""
        return self._df_err

    @property
    def std_err(self):
        """Standard error of the estimate (SEE).  A scalar.
        For standard errors of parameters, see _se_all, se_alpha, and se_beta.
        """
        return self._std_err

    @property
    def predicted(self):
        """The predicted values of y ('yhat')."""
        return np.squeeze(self._predicted)

    @property
    def resids(self):
        return np.squeeze(self._resids)

    @property
    def jarque_bera(self):
        return np.squeeze(self._jarque_bera)

    @property
    def durbin_watson(self):
        return np.squeeze(self._durbin_watson)

    @property
    def ybar(self):
        """The mean of y."""
        return np.squeeze(self._ybar)

    @property
    def ss_tot(self):
        """Total sum of squares."""
        return np.squeeze(self._ss_tot)

    @property
    def ss_reg(self):
        """Sum of squares of the regression."""
        return np.squeeze(self._ss_reg)

    @property
    def ss_err(self):
        """Sum of squares of the residuals (error sum of squares)."""
        return np.squeeze(self._ss_err)

    @property
    def rsq(self):
        """The coefficent of determination, R-squared."""
        return np.squeeze(self._rsq)

    @property
    def rsq_adj(self):
        """Adjusted R-squared."""
        return np.squeeze(self._rsq_adj)

    @property
    def ms_err(self):
        """Mean squared error the errors (residuals)."""
        return np.squeeze(self._ms_err)

    @property
    def ms_reg(self):
        """Mean squared error the regression (model)."""
        return np.squeeze(self._ms_reg)

    @property
    def fstat(self):
        """F-statistic of the fully specified model."""
        return np.squeeze(self._fstat)

    @property
    def fstat_sig(self):
        """p-value of the F-statistic."""
        return np.squeeze(self._fstat_sig)

    @property
    def se_alpha(self):
        """Standard errors (SE) of the intercept (alpha) only."""
        return self._se_alpha

    @property
    def se_beta(self):
        """Standard errors (SE) of the parameters, excluding the intercept."""
        return self._se_beta

    @property
    def tstat_alpha(self):
        """The t-statistic of the intercept (alpha)."""
        return self._tstat_alpha

    @property
    def tstat_beta(self):
        """The t-statistics of the parameters, excl. the intecept."""
        return self._tstat_beta

    @property
    def pvalue_alpha(self):
        """Two-tailed p values for t-stats of the intercept only."""
        return self._pvalue_alpha

    @property
    def pvalue_beta(self):
        """Two-tailed p values for t-stats of parameters, excl. intercept."""
        return self._pvalue_beta

    @property
    def condition_number(self):
        """Condition number of x; ratio of largest to smallest eigenvalue."""
        return self._condition_number


class Correlation:
    def __init__(self, x, y, window=None, extend_window=True):
        self.x = x
        self.y = y
        if window is not None:
            if extend_window:
                self.x, self.y = _extend_xy(self.x, self.y, window)
            self.x = _swap_rolling_windows(self.x, window=window)
            self.y = _swap_rolling_windows(self.y, window=window)

    @property
    def _pearson(self):
        """
        (Σ(xi - xm)(yi - ym)) / (sqrt(Σ(xi - xm) ^ 2) * sqrt(Σ(yi - ym) ^ 2)
        """
        xm = np.mean(self.x, axis=-1)
        xd = self.x - np.expand_dims(xm, axis=-1)
        ym = np.mean(self.y, axis=-1)
        yd = self.y - np.expand_dims(ym, axis=-1)
        
        eye = np.eye(xd.shape[-1])
        xd1 = np.apply_along_axis(lambda r: np.multiply(r, eye), -1, xd)
        yd1 = np.expand_dims(yd, axis=-1)
        xy = np.matmul(xd1, yd1)
        xy = np.squeeze(xy, axis=-1)
        xy = np.sum(xy, axis=-1)
        
        xstd = np.sqrt(np.sum(np.square(xd), axis=-1))
        ystd = np.sqrt(np.sum(np.square(yd), axis=-1))
        corr = xy / (xstd * ystd)
        return corr
    
    @property
    def _spearman(self):
        """
        1 - 6 * (Σ(xri - yri) ^ 2) / (n * (n ^ 2 - 1))
        """
        xr = np.apply_along_axis(lambda r: _rank_array(r), -1, self.x)
        yr = np.apply_along_axis(lambda r: _rank_array(r), -1, self.y)
        n = self.x.shape[-1]
        sub = xr - yr
        sub = np.square(sub)
        sub = np.sum(sub, axis=-1)
        corr = 1 - 6 * sub / (n * (n ** 2 - 1))
        return corr

    # "Public" results
    @property
    def pearson(self):
        return np.squeeze(self._pearson)

    @property
    def spearman(self):
        return np.squeeze(self._spearman)


class FactorRegression:
    def __init__(self, x_data, y_data, direction="vertical", weights=None):
        """
        可能的输入格式
        type1: x_data和y_data都是stocks为列，dates为行
        type2: x_data是多个factors/stocks为列，dates为行
               y_data是一列y值，dates为行
               
        direction只在type1格式下有效
        direction = 'vertical' 按列对x_data, y_data切片，即取一只股票在日期序列上的数据计算回归
        direction = 'horizontal' 按行对x_data, y_data切片，即取一个时间节点上多只股票的数据计算回归
        
        weights权重 可以是和dates等长的1d数组，在type1情况下也可以是和x_data形状一致的2d矩阵
        """

        if type(x_data) == pd.core.frame.DataFrame or type(x_data) == pd.core.series.Series:
            x_data = x_data.values
        if type(y_data) == pd.core.frame.DataFrame or type(y_data) == pd.core.series.Series:
            y_data = y_data.values
        if type(weights) == pd.core.frame.DataFrame or type(weights) == pd.core.series.Series:
            weights = weights.values
        if not type(x_data) == np.ndarray or not type(y_data) == np.ndarray:
            raise Exception("x_data、y_data不是np.ndarray")
        if x_data.shape[0] != y_data.shape[0]:
            raise Exception("x_data、y_data行数不一致")

        self.x_data = x_data
        self.y_data = y_data
        self.direction = direction
        self.weights = weights
        
        self.size = 0  # direction = 'vertical'时，是date的长度
        self.slices = 1  # 切片数 direction = 'vertical'时，即stock的个数
        self.input_type = ""  # 可能的输入格式
        self.x_ls = None
        self.y_ls = None
        
        if x_data.shape == y_data.shape:
            self.input_type = "type1"
            if self.direction == "vertical":
                self.size = x_data.shape[0]
                self.slices = x_data.shape[1]
                self.x_data = np.transpose(self.x_data)  # 垂直切片时执行转置
                self.y_data = np.transpose(self.y_data)
            elif self.direction == "horizontal":
                self.size = x_data.shape[1]
                self.slices = x_data.shape[0]
            else:
                raise Exception("direction只能是vertical或horizontal")
        elif y_data.ndim == 1 or y_data.shape[1] == 1:
            self.input_type = "type2"
            self.size = x_data.shape[0]
            self.slices = 1
        else:
            raise Exception("x_data、y_data输入格式无效")

    def prepare_ls(self):
        # 统一x、y维度  0 stock 1 date 2 data
        if self.input_type == "type1":  # 添加axis=2
            self.x_ls = np.atleast_3d(self.x_data)
            self.y_ls = np.atleast_3d(self.y_data)
        elif self.input_type == "type2":
            self.x_ls = np.atleast_2d(self.x_data)[None, :, :]
            self.y_ls = np.atleast_2d(self.y_data)[None, :, :]  # 添加axis=2，axis=0
        
        if self.weights is None:
            pass
        # 对所有切片使用同一组权重
        elif isinstance(self.weights, list) or (type(self.weights) == np.ndarray and self.weights.ndim == 1):
            self.weights = list(self.weights)
            if len(self.weights) != self.size:
                raise Exception("weights和x_data长度不一样")
            weights = np.atleast_2d(self.weights)
            self.weights = weights.repeat(self.slices, axis=0)  # 在axis=1上升维，重复切片次
        # 权重和x,y形状一致
        elif self.input_type == "type1" and type(self.weights) == np.ndarray:
            if self.direction == "vertical":
                self.weights = self.weights.transpose()
            if self.weights.shape != self.x_data.shape:
                raise Exception("weights输入格式无效")
        else:
            raise Exception("weights输入格式无效")
        
        # print("fr:\n", self.x_ls, "\n\n", self.y_ls, "\n\n")  # 无论哪个type，xy都会转化成3维
        # print("fr:\n", self.weights, "\n\n")  # 无论哪个type，w都会转化成2维
    
    def get_ols(self, window=None, extend_window=True, has_const=False, use_const=True):
        self.prepare_ls()
        return LeastSquares(x=self.x_ls, y=self.y_ls,
                            window=window, extend_window=extend_window,
                            has_const=has_const, use_const=use_const)

    def get_wls(self, window=None, extend_window=True, has_const=False, use_const=True):
        self.prepare_ls()
        return LeastSquares(x=self.x_ls, y=self.y_ls, w=self.weights,
                            window=window, extend_window=extend_window,
                            has_const=has_const, use_const=use_const)

    def get_corr(self, window=None, extend_window=True):
        if self.input_type != "type1":
            raise Exception("相关性只能支持type1类型输入")
        return Correlation(x=self.x_data, y=self.y_data,
                           window=window, extend_window=extend_window)