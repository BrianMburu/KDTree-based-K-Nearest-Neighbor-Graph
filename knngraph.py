import numpy as np
from sklearn.neighbors import KDTree
import time

class KNNGraph:
    """
    KDTree-based K-Nearest Neighbor graph implementation
    """
    def __init__(self, known_embeddings, known_labels):
        """
        Initialize the KDTree with known embeddings and labels
        """
        self.known_embeddings = known_embeddings
        self.known_labels = known_labels
        self.tree = KDTree(known_embeddings)
        
    def get_closest_neighbour(self, query_embedding):
        """
        Returns the closest neighbor and its distance for a given query embedding
        """
        distances, indices = self.tree.query(query_embedding.reshape(1, -1), k=1)
        closest_label = self.known_labels[indices[0][0]]
        closest_distance = distances[0][0]
        
        return closest_label, closest_distance
    
    def test_performance(self, testX, testy):
        """
        Tests the performance of the KNN graph on a given set of test embeddings and labels
        """
        mis_labeled=0
        start_time = time.time()
        
        for i, x in enumerate(testX):
            closest_label, closest_distance = self.get_closest_neighbour(x)
            if testy[i] != closest_label:
                mis_labeled+=1

        end_time = time.time()
        total_time = end_time - start_time
        accuracy = 100 - (mis_labeled / len(testy)) * 100
        
        return accuracy, mis_labeled, total_time
