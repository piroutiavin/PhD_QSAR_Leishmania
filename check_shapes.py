import pandas as pd
X_tr = pd.read_csv('data/processed/X_train.csv', index_col=0, nrows=2)
X_te = pd.read_csv('data/processed/X_test.csv', index_col=0, nrows=2)
y_tr = pd.read_csv('data/processed/y_train.csv', index_col=0)
y_te = pd.read_csv('data/processed/y_test.csv', index_col=0)
print(f"X_train cols: {X_tr.shape[1]}, y_train rows: {len(y_tr)}")
print(f"X_test cols: {X_te.shape[1]}, y_test rows: {len(y_te)}")
ac = y_tr['activity_class']
print(f"Train active: {(ac=='Active').sum()}, inactive: {(ac=='Inactive').sum()}")
ac2 = y_te['activity_class']
print(f"Test active: {(ac2=='Active').sum()}, inactive: {(ac2=='Inactive').sum()}")
