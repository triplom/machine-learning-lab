# Regression Cheatsheet

## Quick Import

```python
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
```

## Algorithm Selection

| Situation | Algorithm |
|-----------|-----------|
| 1 feature, linear | Simple Linear Regression |
| Many features, linear | Multiple Linear Regression |
| Non-linear, smooth curve | Polynomial Regression |
| Non-linear, robust | SVR (RBF kernel) |
| Non-linear, interpretable | Decision Tree |
| Non-linear, best accuracy | Random Forest |

## Common Code Patterns

### Linear Regression
```python
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.coef_, model.intercept_)
```

### Polynomial Regression
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
# Predict new value
model.predict(poly.transform([[6.5]]))
```

### SVR (scale first!)
```python
sc_X = StandardScaler(); sc_y = StandardScaler()
X_sc = sc_X.fit_transform(X)
y_sc = sc_y.fit_transform(y.reshape(-1,1)).ravel()
svr = SVR(kernel="rbf").fit(X_sc, y_sc)
pred = sc_y.inverse_transform(svr.predict(sc_X.transform([[6.5]])).reshape(-1,1))
```

### Random Forest
```python
rf = RandomForestRegressor(n_estimators=300, random_state=0)
rf.fit(X_train, y_train)
rf.feature_importances_
```

## Evaluation Metrics

```python
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
```

| Metric | Range | Better |
|--------|-------|--------|
| R² | (-∞, 1] | Closer to 1 |
| RMSE | [0, ∞) | Lower |
| MAE | [0, ∞) | Lower |

## Visualisation

```python
plt.scatter(X_test, y_test, color="red", label="Actual")
plt.plot(X_train, model.predict(X_train), color="blue", label="Regression line")
plt.legend(); plt.show()
```
