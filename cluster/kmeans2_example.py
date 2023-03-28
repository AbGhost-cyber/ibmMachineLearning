import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

cust_df = pd.read_csv('../datasets/Cust_Segmentation.csv')

# PREPROCESSING
df = cust_df.drop(columns='Address')
# : indicates that we want to select all the rows
# while `1:` indicates that we only want to select the columns
# starting from the second column (index 1) up to the end of the array.
X = df.values[:, 1:]

if __name__ == '__main__':
    print(cust_df.head(n=2))
    print(X[0:2])
