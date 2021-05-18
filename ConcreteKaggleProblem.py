# -*- coding: utf-8 -*-
#%%
# EDA
    # 
#%%
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
sns.set(color_codes=True)
from scipy import stats
import numpy as np
#%%
conc = pd.read_csv("Concrete_Data_Yeh.csv")

#%%
cols = list(conc.columns)
cols.pop()
#['cement', 'slag', 'flyash', 'water', 'superplasticizer', 'coarseaggregate', 'fineaggregate', 'age', 'csMPa']

#%%
conc.dtypes

#%%

conc['age'] = conc['age'].astype('float')
#%%
conc.isnull().sum()

#%%
desc = conc.describe()

#%%
sns.distplot(conc['csMPa'])

#%%
fig, axs = plt.subplots(ncols=8)
sns.distplot(conc['age'], ax=axs[0])
sns.distplot(conc['cement'], ax=axs[0])

cols = list(zip( cols, range(0,8)))

fig, axs = plt.subplots(ncols=len(cols))

for f, a in cols:
    sns.distplot(conc[f], ax=axs[a])

fig.set_size_inches(12, 12)
fig.savefig('mats_distro.png')

#%%
#corr = conc.iloc[:,:-1].corr() 
corr = conc.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, annot=True, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
#%%
sns.scatterplot( x=conc['age'], y=conc['csMPa'])

#%%
y=conc['csMPa']
sns.scatterplot( x=conc['cement'], y=y)

#%%
sns.scatterplot( x=conc['superplasticizer'],y=y)
#%%

sns.scatterplot( x=conc['age'],y=y, label='age')
sns.scatterplot( x=conc['cement'], y=y, label='cement')
sns.scatterplot( x=conc['superplasticizer'], y=y, label='superplasticizer')

plt.xlabel('Independent Features')
plt.title('Relationship of highly correlated features with csMPA')
fig.legend()
plt.show(fig)
#%%

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
import statsmodels.formula.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

#%%

std_scaler = StandardScaler()

#%%

X, y = conc.iloc[:,:-1], conc['csMPa']
X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
)

X_train_std = std.fit_transform(X_train)
X_test_std = std.transform(X_test)
#%%

lm = sm.OLS( y_train, X_train_std).fit()
lm.summary()

#%%
from sklearn.metrics import mean_squared_error

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Training set")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)

#%%
    
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y)
#%%

polynomial_features= PolynomialFeatures(degree=2)
X_train_poly = polynomial_features.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train_poly, y_train)
y_poly_pred = model.predict(X_train_poly)

#%%

rmse = np.sqrt(mean_squared_error(y_train,y_poly_pred))
r2 = r2_score(y_train,y_poly_pred)
print(rmse)
print(r2)

#%%



poly_range = list(range(1, 10))
poly_scores = []

for p in poly_range:
    poly_feat = PolynomialFeatures(degree=p)
    
    polynomial_reg = Pipeline(( 
        ("std_scaler", std_scaler),
        ("poly_feat", poly_feat),
        ("lin_reg", LinearRegression())
    ))
    
    scores = cross_val_score(polynomial_reg, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    poly_scores.append((p,scores.mean()))

print(poly_scores)
#%%

poly_scores = []
param_grid = {'poly_feat__degree': list(range(1,10))}

polynomial_reg = Pipeline(( 
    ("std_scaler", std_scaler),
    ("poly_feat", poly_feat),
    ("lin_reg", LinearRegression())
))

grid = GridSearchCV(polynomial_reg, param_grid, cv=10, scoring='neg_mean_squared_error')

grid.fit(X_train, y_train)

#%%

print("All GRID results\n-----------------------------------------------")
cvres = grid.cv_results_
#cvres is a results data structure. PRINT it!  print(cvres)
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print("GRID", mean_score, params)
print("\nAll GRID results\n-----------------------------------------------")
#print("grid_search.cv_results_", grid_search.cv_results_)
#estimator : estimator object. This is assumed to implement the scikit-learn estimator interface.  
#            Either estimator needs to provide a score function, or scoring must be passed.
#Accuracy is the default for classification; feel free to change this to precision, recall, fbeta
print("Best score: %0.3f" % grid.best_score_)
print("Best parameters set:")
best_parameters = grid.best_estimator_.get_params()
for param_name in sorted(grid.best_params_.keys()):
    print("\t%s: %r" %(param_name, best_parameters[param_name]))

#%%

poly_model = grid.best_estimator_
#%%

train_preds = poly_model.predict(X_train)
test_preds = poly_model.predict(X_test)
#%%

train_acc = np.sqrt(mean_squared_error(train_preds, y_train))
test_acc = np.sqrt(mean_squared_error(test_preds, y_test))
    
print(train_acc, test_acc)