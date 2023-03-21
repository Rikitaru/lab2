import pandas as pd
df_AAPL = pd.read_csv(r"D:\3\AAPL.csv", index_col = "Date")
print(df_AAPL)
from sklearn.preprocessing import StandardScaler
data = df_AAPL.drop('Adj Close', 1)
sc = StandardScaler()
data = sc.fit_transform(data)


from sklearn.decomposition import PCA
pca = PCA()
data = pca.fit_transform(data)
print(data)


explained_variance = pca.explained_variance_ratio_
print(explained_variance)