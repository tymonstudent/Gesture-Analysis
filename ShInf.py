import numpy as np

def compute_information(total_count, cluster_counts):
    """
    Computes the information value for each gesture based on the provided counts.

    Parameters:
    total_count (int): The total number of gestures.
    cluster_counts (list of int): The counts of gestures in each cluster.

    Returns:
    dict: A dictionary containing the information values for each cluster.
    """
    information_values = {}

    for i, count in enumerate(cluster_counts):
        # Calculate probability p(x)
        p_x = count / total_count if total_count > 0 else 0
        
        # Calculate information value I(x)
        if p_x > 0:
            I_x = -np.log2(p_x)
        else:
            I_x = 0  # I(x) is 0 if p(x) is 0
        
        information_values[f'Cluster {i}'] = I_x

    return information_values

# Example usage:
total = 53  # Total number of gestures
clusters = [11,18,1,15,7]  # Example counts for each cluster
info_values = compute_information(total, clusters)
print(info_values)
