import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


#load dataset 
housing= pd.read_csv('housing.csv')
print(housing)
housing.info()
print(housing.describe())

#check the ocean proximity
print(housing["ocean_proximity"].value_counts())

#CREATE TEST SET

# create categories for median income category 1 ranges from 0 to 1.5 (i.e., less than $15,000), category 2 from 1.5 to 3, and so on
housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#VISUALIZATION

housing = strat_train_set.copy()
# #scatter plot
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#  s=housing["population"]/100, label="population", figsize=(10,7),
#  c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
# plt.legend()
# plt.show()

#interactive scatter plot
fig = px.scatter(
    housing,
    x="longitude",
    y="latitude",
    size="population",
    color="median_house_value",
    color_continuous_scale=px.colors.sequential.Jet,
    labels={"population": "Population", "median_house_value": "Median House Value"},
    title="Housing Data",
    size_max=15,
    opacity=0.4,
    width=1000,
    height=700
)
fig.update_layout(
    legend_title_text="Population",
    coloraxis_colorbar=dict(title="Median House Value")
)
fig.show()


#CORRELATION

housing_corr=strat_train_set.copy()
# Drop income_cat column
housing_corr = housing_corr.drop(columns=['income_cat'])
# One-hot encoding for ocean_proximity
housing_encoded = pd.get_dummies(housing_corr, columns=['ocean_proximity'], drop_first=True)

#checking the correlation between median house value and the other variables
corr_matrix = housing_encoded.corr()
# Sorting correlations with median_house_value
corr_sorted = corr_matrix["median_house_value"].sort_values(ascending=False)
corr_sorted = corr_sorted.reset_index()
corr_sorted.columns = ['Feature', 'Correlation']


#interactive bar plot
fig = go.Figure(go.Bar(
    x=corr_sorted['Correlation'],
    y=corr_sorted['Feature'],
    orientation='h',
    marker=dict(color=corr_sorted['Correlation'], colorscale='Viridis')
))
fig.update_layout(
    title='Correlation of Features with Median House Value',
    xaxis_title='Correlation Coefficient',
    yaxis_title='Features',
    yaxis=dict(autorange='reversed'),  
    height=800,
    template='plotly_white'
)
fig.show()

# Create an interactive scatter plot only for the biggest correlation with median income
fig = px.scatter(
    housing,
    x="median_income",
    y="median_house_value",
    color="median_house_value",  # Color by median house value
    color_continuous_scale=px.colors.sequential.Viridis,  # Color scale
    size="population",  # Optional: Size by population
    size_max=20,  # Maximum marker size
    opacity=0.5,  # Marker opacity
    labels={"median_income": "Median Income", "median_house_value": "Median House Value"},
    title="Interactive Scatter Plot of Median Income vs. Median House Value"
)
fig.update_layout(
    xaxis_title="Median Income",
    yaxis_title="Median House Value",
    template="plotly_white",
    height=700
)
fig.show()

#PREPARING THE DATA
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

#take care of missing values 
#replace each attribute’s missing values with the median of that attribute
imputer = SimpleImputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)
imputer.statistics_
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

#handling categorical variables
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot.toarray()

#attributes adder
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): 
            self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
         return self 
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
             bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
             return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
             return np.c_[X, rooms_per_household, population_per_household]
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

#pipeline for missing values, attributes adder and scaling
num_pipeline = Pipeline([
 ('imputer', SimpleImputer(strategy="median")),
 ('attribs_adder', CombinedAttributesAdder()),
 ('std_scaler', StandardScaler()),
 ])
housing_num_tr = num_pipeline.fit_transform(housing_num)

#pipeline for missing values and for categorical attributes
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
 ("num", num_pipeline, num_attribs),
 ("cat", OneHotEncoder(), cat_attribs),
 ])
housing_prepared = full_pipeline.fit_transform(housing)

#CROSS VALIDATION
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

from sklearn.model_selection import cross_val_score
scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-scores)
print(forest_rmse_scores)

#GRID SEARCH
param_grid = [
 {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
 {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
 ]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
 scoring='neg_mean_squared_error',
return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs

# Combine feature importances with feature names and sort
sorted_features = sorted(zip(feature_importances, attributes), reverse=True)
sorted_importances, sorted_attributes = zip(*sorted_features)

# Create a bar plot with matplotlib
plt.figure(figsize=(10, 6))
plt.barh(sorted_attributes, sorted_importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.gca().invert_yaxis()
plt.show()



#EVALUATION ON TEST SET

final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

#confidence interval
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
    loc=squared_errors.mean(),
    scale=stats.sem(squared_errors)))

# Define the function to calculate MAPE
def mean_absolute_percentage_error(y_true, final_predictions):
    y_true, final_predictions = np.array(y_true), np.array(final_predictions)
    return np.mean(np.abs((y_true - final_predictions) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, final_predictions)
mae = mean_absolute_error(y_test, final_predictions)
mse = mean_squared_error(y_test, final_predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, final_predictions)
print(f'MAE: {mae:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'R2: {r2*100:.2f}%')
print(f'MAPE: {mape:.2f}%')

# Create a DataFrame to compare actual and predicted values
comparison_df = pd.DataFrame({'Actual': y_test, 'Predicted': final_predictions})

# Interactive scatter plot
fig = px.scatter(comparison_df, x='Actual', y='Predicted', opacity=0.5, title='Actual vs Predicted Prices')
fig.add_shape(type='line',
              x0=min(y_test), y0=min(y_test), 
              x1=max(y_test), y1=max(y_test),
              line=dict(color='Red',))
fig.show()


