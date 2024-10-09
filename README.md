# Movie collection Prediction Using Decision Tree Regression

![Project Image](project_image.jpg)

## **Features**
- **Handling Missing Data**: Missing values in the dataset (for example, in the 'Time_taken' column) are replaced with the mean value of the column.
- **One-Hot Encoding**: Categorical features such as "3D_available" and "Genre" are encoded using one-hot encoding for use in the machine learning model.
- **Decision Tree Regression**: Different decision tree models are trained and visualized, including variations on depth and minimum samples split/leaf.
- **Model Evaluation**: The performance of the models is evaluated using the mean squared error (MSE) and R-squared (R²) metrics.
- **Visualization**: Decision trees are visualized using the graphviz library to show how the trees split at various nodes based on feature importance.

## **Dataset**
The dataset includes features like:
- **3D Available**: Whether the movie is available in 3D (binary feature).
- **Genre**: Genre of the movie (categorical feature).
- **Time Taken**: Time taken to produce the movie (numeric feature).
- **Collection**: The target variable representing the movie's box office collection (numeric feature).

## **Model**
The project explores different configurations of Decision Tree Regressors:
- **Tree 1**: Decision tree with a maximum depth of 3.
- **Tree 2**: Decision tree with a minimum samples split of 40.
- **Tree 3**: Decision tree with a minimum samples leaf of 25 and a maximum depth of 4.

The trees are trained using a split of 80% for training data and 20% for test data.

## **Metrics**
- **Mean Squared Error (MSE)**: Measures the average of the squared differences between actual and predicted values.
- **R² Score**: Represents the proportion of variance for the dependent variable that's explained by the independent variables.

Example outputs from the models:
- MSE on test data: (calculated using `mean_squared_error`)
- R² score on train and test data: (calculated using `r2_score`)

## **Visualizations**
Three decision tree visualizations are generated using graphviz:
- **Tree 1**: Depth limited to 3.
- **Tree 2**: Minimum samples split set to 40.
- **Tree 3**: Minimum samples leaf set to 25 and maximum depth set to 4.

## **Libraries Used**
- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `sklearn`
- `pydotplus`
- `graphviz`

## **Results**
- **MSE**: `<calculated_value>`
- **R² on Train Data**: `<calculated_value>`
- **R² on Test Data**: `<calculated_value>`

The decision trees help in visualizing how features like "Time Taken" and "Genre" influence the prediction of movie collections.

## **Conclusion**
This project demonstrates the effectiveness of Decision Tree Regression in predicting numeric values and the importance of visualizing decision trees to understand model behavior.
