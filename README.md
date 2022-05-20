[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jorgevazquezperez/Quantum-image-classifier.git/main?labpath=docs%2FUsageTutorial.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Quantum image classifier
## Data use
You can generate synthetic data by calling the function `generate_synthetic_data(n_dim: int, n_clusters: int, m_samples: int)` implemented in *data_generator.py*. You have to be aware that, in order to Nearest Centroid to work, `n_dim` has to be power of 2.
This function returns a set of m_samples vectors X with a set of labels y associated with the vector in the same possition on X. Example:

```{r}
X, y = generate_synthetic_data(8, 4, 250)
train_X = X[:200]
train_y = y[:200]
test_X = X[200:]
test_y = y[200:]
```

If you want, you can also use the MNIST dataset with a PCA function used to reduce the dimension to n components calling `get_MNIST(n_components)` implemented in *data_loader.py*. Same as with the synthetic data, you have to be aware to use only an power of 2 to make Nearest Centroid work. Example:

```{r}
train_X, train_y, test_X, test_y = get_MNIST(8)
```

## Classifiers
### Nearest centroid
Once you get the data, you need to create the object NearestCentroid with the training dataset that you want. After that, you can call the function `predict(self, X: np.ndarray)` owned by the defined object. Example:

```{r}
train_X, train_y, test_X, test_y = get_MNIST(8)
nearest_centroid = NearestCentroid(train_X, train_y, n_dim)
labels_predicted = nearest_centroid.predict(test_X)
```

