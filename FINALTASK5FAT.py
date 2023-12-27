import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from scipy import stats
import ipaddress

# Reading the dataset
df = pd.read_csv("Dataset-Unicauca-Version2-87Atts.csv", nrows=400000)
df = df.sample(replace=False, frac=0.3)

# Check for missing values
def check_empty_cells(dataframe):
    empty_cells = dataframe.isnull().sum().sum()
    if empty_cells > 0:
        print("Empty cells in the DataFrame! WARNING!")
    else:
        print("No empty cells in the DataFrame! GREAT")

check_empty_cells(df)

# Remove servers with IP starting with '10.200.7'
df = df[~df['Destination.IP'].astype(str).str.startswith('10.200.7')]

# Restrict the distribution of labels
lower_label_count = 500
upper_label_count = 15000

few_labels = df.groupby("L7Protocol")["L7Protocol"].count()
few_labels = few_labels[(few_labels < lower_label_count) | (few_labels > upper_label_count)].index.tolist()
df = df[~df['L7Protocol'].isin(few_labels)]

# Function for converting IP addresses to integer values
def ip_to_int(ip):
    return int(ipaddress.IPv4Address(ip))

# Apply the "ip_to_int" function to the "Source.IP" and "Destination.IP" columns
df['Source.IP'] = df['Source.IP'].apply(ip_to_int)
df['Destination.IP'] = df['Destination.IP'].apply(ip_to_int)

# Remove non-informative features
df = df.drop(['Flow.ID','Label', 'Timestamp','ProtocolName','Fwd.Avg.Bytes.Bulk',
              'Fwd.Avg.Packets.Bulk','Fwd.Avg.Bulk.Rate','Bwd.Avg.Bytes.Bulk',
              'Bwd.Avg.Packets.Bulk','Bwd.Avg.Bulk.Rate','CWE.Flag.Count',
              'ECE.Flag.Count','Bwd.PSH.Flags','Fwd.URG.Flags','Bwd.URG.Flags','RST.Flag.Count'], axis=1)

# Normalize the DataFrame
scaler = StandardScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df = df_normalized

# Remove outliers
df_z_score = df_normalized[(np.abs(stats.zscore(df_normalized)) < 3).all(axis=1)]
df = df_z_score

# Define the list of all feature names
column_names = df.columns.tolist()

# Create the Spectral Clustering object
n_clusters = 3  # adjust as needed
spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=0)

# Perform clustering
cluster_labels = spectral.fit_predict(df[column_names])

# Display the clustering results
df['Cluster_Labels'] = cluster_labels
print(df[['L7Protocol', 'Cluster_Labels']])

# Calculate Adjusted Rand Index
ari = adjusted_rand_score(df['L7Protocol'], df['Cluster_Labels'])
print(f"The Adjusted Rand Index (ARI) for Spectral Clustering is: {ari}")

# Visualize the results or perform further analysis as needed
# ...

# Plotting the clusters
for cluster in range(n_clusters):
    plt.scatter(df[df['Cluster_Labels'] == cluster]['Source.IP'], df[df['Cluster_Labels'] == cluster]['Destination.IP'], label=f'Cluster {cluster}')

plt.xlabel('Source.IP')
plt.ylabel('Destination.IP')
plt.title('Spectral Clustering Results')
plt.legend()
plt.show()
