import pandas as pd
file_path='train_data.csv'
df = pd.read_csv(file_path)

import pandas as pd
file_path='train_data.csv'
df = pd.read_csv(file_path)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['condition'] = label_encoder.fit_transform(df['condition'])
import numpy as np

class RandomForestRegressorScratch:
    def __init__(self, n_trees=100, max_depth=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)
        for _ in range(self.n_trees):
            indices = np.random.choice(X.shape[0], X.shape[0], replace=True)
            X_bootstrap = X.iloc[indices]
            y_bootstrap = y.iloc[indices]
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X)
        return np.mean(predictions, axis=1)

X = df.drop(['HR', 'uuid'], axis=1)
y = df['HR']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Mean Squared Error (Random Forest): {mse_rf}')
print(f'R-squared (Random Forest): {r2_rf}')


### Now predictions 

test_data = pd.read_csv('test_data.csv')
label_encoder = LabelEncoder()
test_data['condition'] = label_encoder.fit_transform(test_data['condition'])
test_data_processed = test_data.drop(['uuid'], axis=1)
predictions = rf_model.predict(test_data_processed)
# test_data['Predicted_HR'] = predictions
# test_data.to_csv('sample_output_generated1.csv', index=False)

output_data = pd.DataFrame({'uuid': test_data['uuid'], 'Predicted_HR': predictions})
output_data.to_csv('results.csv', index=False)

