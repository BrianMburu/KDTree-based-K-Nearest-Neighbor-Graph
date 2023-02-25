## KDTree-based K-Nearest Neighbor Graph

### Overview

This code implements a K-Nearest Neighbor graph using KDTree, a fast algorithm for finding nearest neighbors. The graph is initialized with a set of known embeddings and their corresponding labels. The algorithm then constructs a KDTree from the known embeddings to allow for efficient neighbor search. The performance of the graph is tested by querying the graph with a set of test embeddings and measuring the accuracy of the retrieved nearest neighbor labels.

### Dependencies

This code requires the following packages:

- numpy
- sklearn

### Usage

To create a K-Nearest Neighbor graph, simply pass the known embeddings and their corresponding labels to the constructor of the KNNGraph class:

```python
graph = KNNGraph(known_embeddings, known_labels)
```

To get the closest neighbor and its distance for a given query embedding, call the get_closest_neighbour method:

```python
closest_label, closest_distance = graph.get_closest_neighbour(query_embedding)
```

To test the performance of the graph on a set of test embeddings and labels, call the test_performance method:

```python
accuracy, mis_labeled, total_time = graph.test_performance(testX, testy)
```

### Time Complexity

The time complexity of the major operations in this algorithm are as follows:

#### Initialization:

Building the KDTree: O(d*n*log(n)), where d is the dimensionality of the embeddings and n is the number of embeddings.

#### Querying:

Finding the closest neighbor: O(log(n)), where n is the number of embeddings.

#### Testing Performance:

Finding the closest neighbor for each test embedding: O(m\*log(n)), where m is the number of test embeddings and n is the number of known embeddings.

_Note:_ that these time complexities are provided as a rough estimate and may vary depending on the specific implementation and hardware used.

### References:

- "Nearest Neighbor Methods in Learning and Vision: Theory and Practice" by Trevor Hastie, Robert Tibshirani, and Martin Wainwright.
- "Data Structures and Algorithms for Nearest Neighbor Search in General Metric Spaces" by Peter N. Yianilos
- "Efficient K-Nearest Neighbor Graph Construction for Generic Similarity Measures" by Wei Dong, Charikar Moses, and Kai Li.
- "An Introduction to the Analysis of Algorithms" by Robert Sedgewick and Philippe Flajolet.
- "Scikit-learn" documentation on KDTree and k-nearest neighbor algorithms.
