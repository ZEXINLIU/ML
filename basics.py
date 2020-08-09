from pandas import read_csv
from matplotlib import pyplot
from pandas.plotting import scatter_matrix

"""
sepal: 花萼
petal: 花瓣
"""

# creating a Pandas DataFrame
import numpy
import pandas
myarray = numpy.array([[1, 2, 3], [4, 5, 6]])
rownames = ['a', 'b']
colnames = ['one', 'two', 'three']
mydataframe = pandas.DataFrame(myarray, index=rownames, columns=colnames)
print(mydataframe)

# load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# dimension of dataset
print(dataset.shape)

# eyeball the data
print(dataset.head(20))

# description: count, mean, std, min, 25%, 50%, 75%, max
print(dataset.describe())

# class distribution
print(dataset.groupby('class').size())
# sepal-length distribution
print(dataset.groupby('sepal-length').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()

# histgrams, perhaps two of the input variables have a Gaussian distribution
dataset.hist()
pyplot.show()

# scatter plot matrix, create pairwise scatterplots of all attributes.
scatter_matrix(dataset)
pyplot.show()



# Standardize data (0 mean, 1 stdev)
from sklearn.preprocessing import StandardScaler
import pandas
import numpy
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
# separate array into input and output components
X = array[:,0:8]
Y = array[:,8]

"""
Standardize numerical data (e.g. mean of 0 and standard deviation of 1) using the scale and center options.
Normalize numerical data (e.g. to a range of 0-1) using the range option.
Explore more advanced feature engineering such as Binarizing.
"""
# can check rescaledX[:,j].mean() = 0, rescaledX[:,j].std() = 1
scaler = StandardScaler().fit(X)
rescaledX = scaler.transform(X)

# summarize transformed data
numpy.set_printoptions(precision=3)
print(rescaledX[0:5,:])
