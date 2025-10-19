## XAI-DESReg


### Installation 

```bash
pip install xaidesreg
```


### Example 

Loading necessary libraries and dataset:  

```python
import pandas as pd 

# models 
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression 

# metrics 
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo

# dataset
from ucimlrepo import fetch_ucirepo   

# InfoDESReg 
from xaidesreg import DES
```

Loading data: 

```python
abalone = fetch_ucirepo(id=1) 
X = abalone.data.features 
y = abalone.data.targets  

le = LabelEncoder() 
X['Sex'] = le.fit_transform(X['Sex'])  

```

Split the dataset into training, validation for DES (DSEL) and testing.  
```python
TEST_SIZE = 0.2 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42) 
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.2, random_state=42)   
```

1. Models pool generation 

```python
pool_models = [
    XGBRegressor(random_state=42), 
    RandomForestRegressor(random_state=42), 
    LinearRegression(), 
    KNeighborsRegressor(), 
]
```

2. Train the models (pool): 

```python 
for model in pool_models: 
    model.fit(X_train, y_train)  
```

3. Usage of our library: 

```python
des = DES(pool_regressors=pool_models, 
          k=6, 
          knn_metric='minkowski', 
          metrics='mape', 
          threshold=0.1)

des.fit(X_dsel, y_dsel) 
``` 

4. Testing 

```python 
des.score(X_test, y_test) # based on MSE 
```

5. Explainability 

```python 
index = 47
X_test.iloc[index]

prediction, contribution_df, neighbors_df = des.predict_xai(X_test.iloc[[index]])

print(prediction) 
```

The content of contribution data frame: 
| Model                  | Predictions | Competence | Weights |
|------------------------|-------------|------------|---------|
| XGBRegressor           | 9.133       | 0.064      | 15.609  |
| RandomForestRegressor  | 8.890       | 0.078      | 12.853  |
| LinearRegression       | 8.913       | 0.090      | 11.141  |
| KNeighborsRegressor    | 9.000       | 0.072      | 13.846  |

The samples in the region of competence (neighbors_df): 
| Sex | Length | Diameter | Height | Whole_weight | Shucked_weight | Viscera_weight | Shell_weight | Target |
|-----|--------|----------|--------|--------------|----------------|----------------|--------------|--------|
| 2   | 0.565  | 0.44     | 0.125  | 0.802        | 0.3595         | 0.1825         | 0.215        | 9      |
| 2   | 0.55   | 0.425    | 0.15   | 0.8315       | 0.411          | 0.1765         | 0.2165       | 10     |
| 2   | 0.56   | 0.415    | 0.13   | 0.7615       | 0.3695         | 0.17           | 0.1955       | 8      |
| 2   | 0.545  | 0.41     | 0.12   | 0.793        | 0.434          | 0.1405         | 0.19         | 9      |
| 2   | 0.56   | 0.415    | 0.145  | 0.852        | 0.43           | 0.1885         | 0.205        | 8      |
| 2   | 0.54   | 0.42     | 0.135  | 0.8075       | 0.3485         | 0.1795         | 0.235        | 11     |
