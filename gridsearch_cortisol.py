import subprocess

# Print Python version and install/update/import libraries!
subprocess.run("python -V")

subprocess.run("pip install -U scikit-learn")
import sklearn
print("scikit-learn", sklearn.__version__)

subprocess.run("pip install -U matplotlib")
import matplotlib
print("matplotlib", matplotlib.__version__)

subprocess.run("pip install -U seaborn")
import seaborn
print("seaborn", seaborn.__version__)

subprocess.run("pip install -U polars")
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

N_NEIGHBORS_OPTIONS = [int for int in range(1, 1000)]
WEIGHTS_OPTIONS = ["uniform", "distance"]
ALGORITHM_OPTIONS = ["ball_tree", "kd_tree", "brute"]
LEAF_SIZE_OPTIONS = [int for int in range(1, 1000)]

param_grid = {
    "kneighborsregressor__n_neighbors" : N_NEIGHBORS_OPTIONS,
    "kneighborsregressor__weights" : WEIGHTS_OPTIONS,
    "kneighborsregressor__algorithm" : ALGORITHM_OPTIONS,
    "kneighborsregressor__leaf_size" : LEAF_SIZE_OPTIONS
}

gridsearchcv = sklearn.model_selection.GridSearchCV(pipeline, param_grid=param_grid)

results = gridsearchcv.fit(X, y)
print(results)