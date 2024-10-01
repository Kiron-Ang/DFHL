# Print Python version and import libraries!
import subprocess
subprocess.run("python -V")

import sklearn
print("scikit-learn", sklearn.__version__)

import matplotlib
print("matplotlib", matplotlib.__version__)

import seaborn
print("seaborn", seaborn.__version__)

import polars
print("polars", polars.__version__)

# Read in two CSV files, one with cortisol data and one with BDI data
cortisol = polars.read_csv("cortisol.csv")
cortisol = cortisol.drop_nulls()
print("cortisol", cortisol.shape)

bdi = polars.read_csv("bdi.csv")
bdi = bdi.drop_nulls()
print("bdi", bdi.shape)

# Make a pipeline to eliminate data leakage
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.neighbors

pipeline = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.RobustScaler(),
    sklearn.neighbors.KNeighborsRegressor()
)

print(pipeline)

# Create two separate DataFrames to hold input X and target y
X = cortisol[:,1:]
print("X", X.shape)
y = cortisol[:,0]
print("y", y.shape)

# Use GridSearchCV to choose the best parameters
import sklearn.model_selection

N_NEIGHBORS_OPTIONS = [1, 10, 100, 100]

param_grid = {
    "kneighborsregressor__n_neighbors" : N_NEIGHBORS_OPTIONS 
}

gridsearchcv = sklearn.model_selection.GridSearchCV(pipeline, param_grid=param_grid)

best_estimator = gridsearchcv.fit(X, y).best_estimator_
print(best_estimator)