#Auto Insurance Customer Analytics

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans

Auto_df = pd.read_csv("C:/Users/Prasanna/Desktop/AutoInsurance.csv")
Auto_df # Displaying the dataset
Auto_df.info() # Checking for attribute datatype
Auto_df.describe() # EDA

# Normalization of values (Selecting only columns with numerical values)
Auto=Auto_df[['Customer Lifetime Value','Income','Monthly Premium Auto','Months Since Last Claim','Months Since Policy Inception','Number of Open Complaints','Number of Policies','Total Claim Amount']]
Auto

# Normalization function
def norm_func(i):
    x = (i - i.min()) / (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Auto.iloc[:, 0:])
df_norm.describe()

# K-Means Clustering

# Calculating Total Within Sum of Squares
TWSS = []
k = list(range(2, 15))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)

TWSS
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")


# Hierarchical Clustering

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
