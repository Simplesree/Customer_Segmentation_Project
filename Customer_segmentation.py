import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Mall_Customers.csv")

# Convert Gender into numbers
encoder = LabelEncoder()
data["Gender"] = encoder.fit_transform(data["Gender"])

# Select features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Elbow Method
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# Create Final Model
kmeans = KMeans(n_clusters=5, random_state=42)

# Create Cluster Column
data["Cluster"] = kmeans.fit_predict(X)

# Print Customers
print(data[["CustomerID",
            "Annual Income (k$)",
            "Spending Score (1-100)",
            "Cluster"]].head(10))

# Final Segmentation Graph
plt.figure(figsize=(10,6))

plt.scatter(
    data["Annual Income (k$)"],
    data["Spending Score (1-100)"],
    c=data["Cluster"],
    cmap="rainbow"
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    color="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation using KMeans")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()

plt.show()

# Save the output
data.to_csv("Customer_Segments.csv", index=False)

print("Customer Segmentation Completed Successfully!")
# Save the output
data.to_csv("Customer_Segments.csv", index=False)

print("Customer Segmentation Completed Successfully!")

# Number of customers in each cluster
print("\nCustomers in each Cluster:")
print(data["Cluster"].value_counts())

# Average values of each cluster
print("\nAverage values of each Cluster:")
print(data.groupby("Cluster")[["Age",
                               "Annual Income (k$)",
                               "Spending Score (1-100)"]].mean())