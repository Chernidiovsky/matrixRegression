
factorRegression用于批量计算ols、wls、pearson相关系数、spearman相关系数，支持rolling。

和statsmodels相关函数的区别是用numpy实现，可以支持多组数据并行计算。计算效率更高，也需要更大的内存。

支持的输入格式有两种
type1: x_data和y_data都是stocks为列，dates为行；
type2: x_data是多个factors/stocks为列，dates为行，y_data是一列y值，dates为行。
   
direction只在type1格式下有效
direction = 'vertical' 在垂直方向上切片处理，即按列对x_data, y_data切片，每片取一只股票在日期序列上的数据计算回归
direction = 'horizontal' 在水平方向上切片处理，即按行对x_data, y_data切片，每片取一个时间节点上多只股票的数据计算回归

weights是针对wls的权重，可以是和dates等长的1d数组，在type1情况下也可以是和x_data形状一致的2d矩阵。

get_ols和get_wls的参数
window表示rolling的步长，不得大于原数列长度。
extend_window表示是否将rolling后的长度扩展至和rolling前一致，方便数据处理时对齐日期。
假设数列a长度=6，window=4，rolling后a变为3*4矩阵，当extend_window=True时，在a前面补充一个3*4的以np.nan填充的矩阵，将a扩展为6*4，其最高维的长度和原长度一致。

参考自
https://github.com/bsolomon1124/pyfinance
的ols.py