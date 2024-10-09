# Importing necessary libraries
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score
from IPython.display import Image
import pydotplus

# Loading the dataset
df = pd.read_csv("drive/MyDrive/Datasets/Movie_regression.csv", header=0)

# Display the first few rows of the dataset
df.head()

# Display the dataset info (data types, missing values, etc.)
df.info()

# Handle missing values in the 'Time_taken' column by replacing them with the mean
df['Time_taken'].fillna(value=df['Time_taken'].mean(), inplace=True)

# Verifying if missing values are filled
df.info()

# Display the first few rows to verify changes
df.head()

# Encoding categorical variables using one-hot encoding
df = pd.get_dummies(df, columns=["3D_available", "Genre"], drop_first=True)

# Splitting the dataset into features (X) and target variable (y)
x = df.loc[:, df.columns != 'Collection']  # Features
y = df['Collection']  # Target variable

# Display the shapes of the feature and target data
print("Shape of X (features):", x.shape)
print("Shape of y (target):", y.shape)

# Split the data into training and test sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Check the shapes of the train and test splits
print("Train/Test split sizes:")
print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
print(f"y_train: {y_train.shape}, y_test: {y_test.shape}")

# Initialize the Decision Tree Regressor with a max depth of 3
regtree = tree.DecisionTreeRegressor(max_depth=3)

# Train the model on the training data
regtree.fit(x_train, y_train)

# Predict the target for both the training and test sets
y_train_pred = regtree.predict(x_train)
y_test_pred = regtree.predict(x_test)

# Calculate the Mean Squared Error (MSE) for the test set predictions
mse_test = mean_squared_error(y_test, y_test_pred)
print(f"Mean Squared Error (Test Set): {mse_test}")

# Calculate R-squared for both the train and test sets
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)
print(f"R-squared (Train Set): {r2_train}")
print(f"R-squared (Test Set): {r2_test}")

# Visualizing the decision tree
dot_data = tree.export_graphviz(regtree, out_file=None, feature_names=x_train.columns, filled=True)

# Create a graph from the dot data
graph = pydotplus.graph_from_dot_data(dot_data)

# Display the decision tree as an image
Image(graph.create_png())

# Training another Decision Tree with a different criterion (max depth)
regtree1 = tree.DecisionTreeRegressor(max_depth=3)
regtree1.fit(x_train, y_train)

# Visualize the decision tree with filled nodes
dot_data1 = tree.export_graphviz(regtree1, out_file=None, feature_names=x_train.columns, filled=True)
graph1 = pydotplus.graph_from_dot_data(dot_data1)
Image(graph1.create_png())

# Training another Decision Tree with a minimum samples split of 40
regtree2 = tree.DecisionTreeRegressor(min_samples_split=40)
regtree2.fit(x_train, y_train)

# Visualize the tree for min_samples_split=40
dot_data2 = tree.export_graphviz(regtree2, out_file=None, feature_names=x_train.columns, filled=True)
graph2 = pydotplus.graph_from_dot_data(dot_data2)
Image(graph2.create_png())

# Training a Decision Tree with a minimum samples leaf of 25 and max depth of 4
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf=25, max_depth=4)
regtree3.fit(x_train, y_train)

# Visualize this decision tree
dot_data3 = tree.export_graphviz(regtree3, out_file=None, feature_names=x_train.columns, filled=True)
graph3 = pydotplus.graph_from_dot_data(dot_data3)
Image(graph3.create_png())
