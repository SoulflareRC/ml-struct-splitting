import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering 
from sklearn.mixture import GaussianMixture 
MAX_N = 3 

class Grouper:
    # Base class for all groupers 
    def __init__(self, max_N=3):
        self.max_N = max_N
    def assign_groups(self, data):
        '''
        the data here should be a numpy matrix that's F x L, each row is a feature vector 
        '''
        raise NotImplementedError("Subclasses should implement this!")
    def name(self): 
        return self.__class__.__name__ 

def remap_to_contiguous(groups: np.ndarray) -> np.ndarray:
    """
    Remap an array of group numbers to ensure they are contiguous, starting from 0.
    :param groups: A 1D numpy array of group assignments.
    :return: A 1D numpy array of remapped group assignments.
    """
    # Step 1: Find unique group numbers and remap them to contiguous numbers starting from 0
    _, remapped_groups = np.unique(groups, return_inverse=True)

    # Step 2: Return the remapped groups (now they are contiguous)
    return remapped_groups

class RandomGrouper(Grouper):
    # This grouper assigns random group numbers to each field in the struct 
    def assign_groups(self, data: np.ndarray) -> np.ndarray:
        """Assign each row of data to a random group."""
        print("data shape: ", data.shape) 
        n_datapoints = data.shape[0]  # number of rows
        n_clusters = self.max_N if self.max_N < n_datapoints else n_datapoints 
        # if # data points is smaller than max_N, then must limit to # data points 
        groups = np.random.randint(0, n_clusters, size=n_datapoints)  # generate random group numbers
        return groups.reshape(1, -1)  
        #     # Step 2: Find unique group numbers in the order they appear
        # unique_groups, remapped_groups = np.unique(groups, return_inverse=True)

        # # Step 3: The remapped_groups are now contiguous, starting from 0
        # return remapped_groups.reshape(1, -1)  # return as a row vector

class NoSplittingGrouper(Grouper):
    # This grouper assigns all fields in the struct to group 0 (no splitting) 
    def assign_groups(self, data: np.ndarray) -> np.ndarray:
        """Assign all rows to a single group."""
        n = data.shape[0]  # number of rows
        groups = np.zeros(n, dtype=int)  # all rows in group 0
        return groups.reshape(1, -1)  # reshape to n x 1

class HotnessGrouper(Grouper):
    def assign_groups(self, data: np.ndarray) -> np.ndarray:
        """
        This implements the heuristic described in the original paper, where we:
        data is a F x L numpy matrix 
        1. calculate bit vectors of each loop (column) 
            - 1 when row is not 0 
            - 0 when row is 0 
        2. calculate sum of the entire matrix, and re-scale the matrix by dividing by matrix sum 
        3. aggregate columns based with identical bit vectors (sum columns with the same bit vectors)  
        4. sort columns by their averages 
        """
        
        n_datapoint = data.shape[0] 
        n_clusters = min(n_datapoint, self.max_N) 
        if n_datapoint == 1:  # Special case for single data point
            return np.array([0])  # Assign it to group 0 (or any group)
        
        # print("data: \n", data)  
        # print("groupings shape: ", groupin) 
        # groups = np.zeros(data.shape[1], dtype=int)
        groupings = np.full(data.shape[0], -1)  
        # print("data: \n", data) 
        # Step 1: Calculate bit vectors
        bit_vectors = (data != 0).astype(int)  # 1 if not zero, 0 if zero
        

        # Step 2: Rescale the matrix by dividing by the matrix sum
        row_sum = np.sum(data, axis=1,keepdims=True)
        row_sum = np.where(row_sum==0, 1e-10, row_sum) 
        # scaled_data = data / row_sum 
        scaled_data = data 
        # print("scaled data: \n", scaled_data) 
        
        # Step 3: Aggregate columns with identical bit vectors
        # Identify unique bit vectors
        unique_bit_vectors, unique_indices = np.unique(bit_vectors, axis=0, return_inverse=True)
        # print("bit vectors: \n", bit_vectors) 
        # print("unique_bit_vectors: \n", unique_bit_vectors) 
        # print("unique_indices: \n", unique_indices)

        # Initialize an array to hold aggregated columns
        # aggregated_columns = np.zeros((data.shape[0], unique_bit_vectors.shape[1]))
        aggregated_columns = [] 
        # Store the associations between columns and bit vectors
        column_bit_vector_associations = []
        
        g_cnt = len(unique_bit_vectors) 
        column_field_access_association = np.zeros((g_cnt, n_datapoint))  # bit vectors indicating which fields where accessed in each aggregated group,   

        # Aggregate columns by summing the fields that have the same bit vector (accessed in the same loops) 
        for i in range(len(unique_bit_vectors)):
            
            target_bit_vector = unique_bit_vectors[i] 
            # print("target_bit_vector: ", target_bit_vector) 
            # Find all columns that have the same bit vector as the current unique one
            matching_columns = np.where(unique_indices == i)[0]
            # print("matching columns: ", matching_columns) 
            matched_columns = scaled_data[matching_columns]
            # print("matched columns: \n", matched_columns) 
            # Sum those columns and assign the result to the aggregated array
            aggregated_column = np.sum(matched_columns, axis=0)
            # print("aggregated column: \n",aggregated_column) 
            # aggregated_columns[:, i] = aggregated_column 
            aggregated_columns.append(aggregated_column) 
            
            column_field_access_association[i][matching_columns] = 1; 
            # print("column_field_access_association:\n", column_field_access_association) 

        # Convert the list of associations to an array
        # column_bit_vector_associations = np.array(column_bit_vector_associations)
        aggregated_columns = np.array(aggregated_columns) 
        # print("column_bit_vector_associations: \n",column_bit_vector_associations)
        # print("aggregated_columns: \n", aggregated_columns)
        # aggregated_columns = np.where(aggregated_columns==0, 1e-10, aggregated_columns) # avoid div by 0 
        
        aggregated_columns_hotness = np.sum(aggregated_columns, axis=1) / np.sum(aggregated_columns) 
        if np.sum(aggregated_columns) == 0: 
            aggregated_columns_hotness = np.sum(aggregated_columns, axis=1) 
        # print("aggregated_columns_hotness\n", aggregated_columns_hotness)
        
        # Step 5: Re-arrange the columns by hotness (descending order)
        sorted_indices = np.argsort(aggregated_columns_hotness)[::-1]  # Sort indices in descending order of hotness
        
        # print("sorted indices: ", sorted_indices )

        # Re-arrange the columns and bit vector associations based on sorted indices
        sorted_aggregated_columns = aggregated_columns[sorted_indices]
        sorted_field_access_association = column_field_access_association[sorted_indices] 
        
        # print("sorted aggregated columns: \n", sorted_aggregated_columns)
        # print("sorted field access association: \n", sorted_field_access_association) 

        # print("max hotness: ", np.max(aggregated_columns_hotness) )
        level = np.max(aggregated_columns_hotness) / n_clusters  
        # print("level: ", level) 
        thresholds = np.arange(1,n_clusters+1) * level 
        # print("thresholds: ", thresholds) 
        
        # further aggregate rows under each level
        final_aggregated_field_vector = np.zeros((n_clusters, n_datapoint), dtype=int)  
        for i in range(g_cnt):  
            hotness = aggregated_columns_hotness[i] 
            # print("hotness: ", hotness) 
            index = np.searchsorted(thresholds, hotness, side='right') 
            if index>= n_clusters:  
                index = n_clusters-1 
            # print("index: ", index) 
            field_access_vector = sorted_field_access_association[i] 
            # print("field access vector: ", field_access_vector) 
            # print("final_aggregated_field_vector[index]: ", final_aggregated_field_vector[index])
            final_aggregated_field_vector[index] = np.bitwise_or(final_aggregated_field_vector[index], field_access_vector.astype(int))  
        # print("final_aggregated_field_vector: \n", final_aggregated_field_vector)
        
        for i in range(n_clusters): 
            field_vector = final_aggregated_field_vector[i] 
            fields = np.where(field_vector==1) 
            groupings[fields] = i 
            
        groupings = np.where(groupings==-1, n_clusters-1, groupings) 
        
        groupings =  groupings.reshape(1, -1)  # reshape to n x 1
        # print(f"Final groupings: {groupings}")
        return groupings
class KMeansGrouper(Grouper):
    # This grouper assigns all fields in the struct to group 0 (no splitting) 
    def assign_groups(self, data: np.ndarray) -> np.ndarray:
        """Assign rows to group with KMeans clustering."""
        n_datapoint = data.shape[0] 
        n_clusters = min(n_datapoint, self.max_N) 
        if n_datapoint == 1:  # Special case for single data point
            return np.array([0])  # Assign it to group 0 (or any group)
        clusterer = KMeans(n_clusters=n_clusters) 
        labels = clusterer.fit_predict(data) 
        # print("kmeans_labels: ", labels) 
        return labels 
    

class AgglomerativeGrouper(Grouper):
    # This grouper assigns all fields in the struct to group 0 (no splitting) 
    def assign_groups(self, data: np.ndarray) -> np.ndarray:
        """Assign rows to group with Agglomerative(hierarchical) clustering."""
        n_datapoint = data.shape[0] 
        n_clusters = min(n_datapoint, self.max_N)
        print(f"shape: {data.shape} max_N: {self.max_N} n_clusters: {n_clusters}")
        if n_datapoint == 1:  # Special case for single data point
            return np.array([0])  # Assign it to group 0 (or any group)
        clusterer = AgglomerativeClustering(n_clusters=n_clusters) 
        labels = clusterer.fit_predict(data)  
        return labels 
    

class SpectralGrouper(Grouper):
    # This grouper assigns all fields in the struct to group 0 (no splitting) 
    def assign_groups(self, data: np.ndarray) -> np.ndarray:
        """Assign rows to group with Agglomerative(hierarchical) clustering."""
        n_datapoint = data.shape[0] 
        n_clusters = min(n_datapoint, self.max_N)
        if n_datapoint == 1:  # Special case for single data point
            return np.array([0])  # Assign it to group 0 (or any group)
        clusterer = SpectralClustering(n_clusters=n_clusters, affinity="rbf", gamma=1.0) 
        labels = clusterer.fit_predict(data)  
        return labels 
    
    

class GMMGrouper(Grouper):
    # This grouper assigns all fields in the struct to group 0 (no splitting) 
    def assign_groups(self, data: np.ndarray) -> np.ndarray:
        """Assign rows to group with Agglomerative(hierarchical) clustering."""
        n_datapoint = data.shape[0] 
        n_clusters = min(n_datapoint, self.max_N)
        if n_datapoint == 1:  # Special case for single data point
            return np.array([0])  # Assign it to group 0 (or any group)
        clusterer = GaussianMixture(n_components=n_clusters) 
        labels = clusterer.fit_predict(data)  
        return labels 


GROUPERS = [
    NoSplittingGrouper, 
    RandomGrouper, 
    HotnessGrouper, 
    KMeansGrouper, 
    AgglomerativeGrouper, 
    SpectralGrouper, 
    GMMGrouper,
]
GROUPERS_CNT = len(GROUPERS) 

def get_all_grouper_names(): 
    return [grouper_class.__name__ for grouper_class in GROUPERS]

def get_all_groupers(max_N=MAX_N):
    return [grouper_class(max_N=max_N) for grouper_class in GROUPERS]


# Example usage
if __name__ == "__main__":
    print(get_all_grouper_names()) 
    
    # Creating a sample n x m numpy array (e.g., 5 rows, 3 columns)
    # data = np.array([[1, 2, 0],
    #                  [4, 0, 0],
    #                  [7, 3, 0],
    #                  [10, 11, 0],
    #                  [13, 14, 4000]])
    
    
    # grouper = GMMGrouper(max_N=3) 
    # groups = grouper.assign_groups(data) 
    # print("Groups: \n", groups)
    
    # grouper = SpectralGrouper(max_N=3) 
    # groups = grouper.assign_groups(data) 
    # print("Groups: \n", groups)
    
    # grouper = AgglomerativeGrouper(max_N=3) 
    # groups = grouper.assign_groups(data) 
    # print("Groups: \n", groups)
    
    # kmeans_grouper = KMeansGrouper(max_N=3) 
    # kmeans_groups = kmeans_grouper.assign_groups(data) 
    # print("Kmeans Groups: \n", kmeans_groups)
    
    # hotness_grouper = HotnessGrouper(max_N=3) 
    # hotness_groups = hotness_grouper.assign_groups(data) 
    # print("Hotness Groups: \n", hotness_groups) 
    
    # # Using RandomGrouper
    # random_grouper = RandomGrouper(max_N=3)
    # random_groups = random_grouper.assign_groups(data)
    # print("Random Groups:\n", random_groups)

    # # Using NoSplittingGrouper
    # no_splitting_grouper = NoSplittingGrouper(max_N=1)
    # no_split_groups = no_splitting_grouper.assign_groups(data)
    # print("No Splitting Groups:\n", no_split_groups)
