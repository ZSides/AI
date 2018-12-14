import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('/home/micha/primery/glc.csv',  header=None,  sep=',')
df.columns =  ['Year',  'Total',  'Gas Fuel',  'Liquid Fuel', 'Solid Fuel', 'Cement', 'Gas Flaring', 'Per Capita']
df.head()
#Year 	Total 	Gas Fuel 	Liquid Fuel 	Solid Fuel 	Cement 	Gas Flaring 	Per Capita



import matplotlib.pyplot as pit 
import seaborn as sns
sns.set(style='white',  context='notebook') 
cols = ['Year', 'Total', 'Gas Fuel', 'Per Capita']
sns.pairplot(df[cols],  size=2.5) 
plt.show()

#cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV'] 
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.5)
hm = sns.heatmap(cm,
	cbar=True,
	annot=True,
	square=True, 
	fmt='.2f',
	annot_kws={'size':  15},
	yticklabels=cols,
	xticklabels=cols)
plt.show()

x = df[['Year']].values 
y = df['Gas Fuel'].values
slr = LinearRegression()
slr.fit (x,  y)
print('Наклон: %.3f'  % slr.coef_[0])
print('Пересечение: %.3f' % slr.intercept_)
price = slr.predict(2007.0)
print('Количество: %.3f' % price)

plt.scatter(x, y, c='blue')
plt.plot(x, slr.predict(x), color='red')
plt.show()

from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),
	max_trials=100,
	min_samples=50,
	residual_metric=lambda x: np.sum(np.abs(x), axis=1),
	residual_threshold=50.0,
	random_state=0)
ransac.fit (x, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

line_X = np.arange(1950, 2010, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])
plt.scatter(x[inlier_mask], y[inlier_mask],
c='blue', marker='o', label='Не-выбросы')
plt.scatter(x[outlier_mask], y[outlier_mask],
c='lightgreen', marker='s', label='Выбросы')
plt.plot(line_X, line_y_ransac, color='red')
plt.show()
price1 = ransac.predict(2007.0)
print('Количество: %.3f' % price1)

