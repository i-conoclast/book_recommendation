from prince import FAMD
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def pre_clustering(data, type="account", 
                   account_id_column=None,
                   account_date_column=None,
                   product_id_column=None):
    if type == "account":
        mms = MinMaxScaler()

        gen_col = [col for col in data.columns if col.startswith('gender')]
        add_col = [col for col in data.columns if col.startswith('add')]
        age_col = [col for col in data.columns if col.startswith('age')]

        data['gender'] = data[gen_col].idxmax(axis=1).str[-1]
        data['address_is'] = data[add_col].idxmax(axis=1).str[-1]
        data['age'] = data[age_col].idxmax(axis=1).str[-1]

        flo_col = [col for col in data.columns if data[col].dtype == float]

        data[flo_col] = mms.fit_transform(data[flo_col])
        data.drop(columns=gen_col+add_col+age_col+[account_id_column, account_date_column], axis=1)

    if type == "product":
        cat_col = [col for col in data.columns if col.startswith('category')]
        pri_col = [col for col in data.columns if col.startswith('shop')]
        pub_col = [col for col in data.columns if col.startswith('pub_')]

        data['category_id'] = data[cat_col].idxmax(axis=1).str[-2:]
        data['pub'] = data[pub_col].idxmax(axis=1).str[-1]
        data['price'] = data[pri_col].idxmax(axis=1).str[-1]

        data.drop(columns=cat_col+pri_col+pub_col+[product_id_column], axis=1)
        
        return data


def famd_kmeans_clustering(data, cluster_column, variance_threshold=0.8, max_clusters=10, random_state=None):
    """
    Dynamically determines n_components for FAMD based on explained variance threshold,
    then performs clustering using K-Means and assigns cluster labels to the original data.

    Parameters:
        data (pd.DataFrame): Input data with mixed categorical and numerical variables.
        variance_threshold (float): Minimum cumulative explained variance to determine n_components.
        max_clusters (int): Maximum number of clusters to evaluate.
        random_state (int, optional): Random seed for reproducibility.
    
    Returns:
        pd.DataFrame: Original DataFrame with an added 'cluster' column.
    """
    # Step 1: Initialize FAMD and fit the data to determine n_components
    famd = FAMD(n_components=min(data.shape[1], 10), random_state=random_state)
    famd = famd.fit(data)
    
    # Calculate cumulative explained variance
    cumulative_variance = famd.explained_inertia_.cumsum()
    
    # Determine n_components dynamically
    n_components = next(i + 1 for i, var in enumerate(cumulative_variance) if var >= variance_threshold)
    print(f"Selected n_components based on variance threshold ({variance_threshold}): {n_components}")
    
    # Step 2: Apply FAMD with the determined n_components
    famd = FAMD(n_components=n_components, random_state=random_state)
    famd_result = famd.fit_transform(data)
    
    best_score = -1
    best_n_clusters = 0
    best_labels = None
    
    # Step 3: Determine the optimal number of clusters using silhouette score
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        labels = kmeans.fit_predict(famd_result)
        
        # Compute silhouette score
        score = silhouette_score(famd_result, labels)
        
        if score > best_score:
            best_score = score
            best_n_clusters = n_clusters
            best_labels = labels
    
    # Step 4: Assign the best cluster labels to the original data
    data_with_clusters = data.copy()
    data_with_clusters[cluster_column] = best_labels
    
    print(f"Optimal number of clusters: {best_n_clusters}")
    print(f"Silhouette score: {best_score:.4f}")
    
    return data_with_clusters
