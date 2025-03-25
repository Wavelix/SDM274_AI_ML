import numpy as np

class DecisionTree:
    # Create an internal class called Node
    class Node:
        def __init__(self) -> None:
            # When a node a leaf node, then we use the output label as the value
            self.value = None
            # When a node is an internal node, then we use feature_index on it
            self.feature_index = None
            # Use a dictionary called children to indicate the children nodes, which contain {feature_value: node, }
            self.children ={}

        def __str__(self) -> str:
            if self.children:
                s = f'Internal node <{self.feature_index}>:\n'
                for fv, node in self.children.items():
                    ss = f'[{fv}]-> {node}'
                    s += '\t' + ss.replace('\n', '\n\t') + '\n'
            else:
                s = f'Leaf node ({self.value})'
            
            return s
    
    def __init__(self, gain_threshhold = 1e-2) -> None:
        # Set a threshold for information gain
        self.gain_threshold = gain_threshhold

    def _entropy(self, y):
        # Compute entropy of output -sum(p(Y=y)log2(p(Y=y))), which is a scalar

        count_y = np.bincount(y)
        prob_y = count_y[np.nonzero(count_y)] / y.size
        entropy_y = -np.sum(prob_y * np.log2(prob_y))
        return entropy_y
    
    def _conditional_entropy(self, feature, y):
        # Compute the conditional entropy according to the formula H(Y|feature) = Sum_{feature_value} p(feature = feature_value) H(Y|feature=feature_value)
        # The arugment feature represents the input data vector of one specific feature
        feature_values = np.unique(feature)
        h = 0.
        for v in feature_values:
            y_sub = y[feature == v]
            prob_y_sub = y_sub.size / y.size
            h += prob_y_sub * self._entropy(y_sub)

        return h
    
    def _information_gain(self, feature, y):
        ig_feature = self._entropy(y) - self._conditional_entropy(feature, y)
        return ig_feature
    
    def _select_feature(self, X, y, features_list):
        # Select the feature with the largest information gain
        if features_list:
            gains = np.apply_along_axis(self._information_gain, 0, X[:, features_list], y)
            index = np.argmax(gains)
            if gains[index] > self.gain_threshold:
                return index

        return None

    def _build_tree(self, X, y, features_list):
        # Build a decision tree recuresively. 
        # The default output should be the label with the maximum counting
        node = DecisionTree.Node()
        labels_count = np.bincount(y) 
        node.value = np.argmax(np.bincount(y))

        # Check whether the labels are the same
        if np.count_nonzero(labels_count) !=1:
            # Select the feature with the largest information gain
            index = self._select_feature(X, y, features_list)

            if index is not None:
                # Remove this feature from the features list
                node.feature_index = features_list.pop(index)

                # Divide the training set according to this selected feature
                # Then use the subset of training examples in each branch to create a sub-tree
                feature_values = np.unique(X[:, node.feature_index])
                for v in feature_values:
                    # Obtain the subset of training examples
                    idx = X[:, node.feature_index] == v
                    X_sub, y_sub = X[idx], y[idx]

                    # Build a sub-tree
                    node.children[v] = self._build_tree(X_sub, y_sub, features_list.copy())

        return node 
    
    
    def train(self, X_train, y_train):
        _, n = X_train.shape
        self.tree_ = self._build_tree(X_train, y_train, list(range(n)))

    def _predict_one(self, x):
        node = self.tree_
        while node.children:
            child = node.children.get(x[node.feature_index])
            if not child:
                break
            node = child
        
        return node.value

    def predict(self, X):
        return np.apply_along_axis(self._predict_one, axis=1, arr=X)
    
    def __str__(self):
        if hasattr(self, 'tree_'):
            return str(self.tree_)
        return ''




if __name__ == '__main__':

    data = np.loadtxt('./lenses/lenses.data', dtype=int)
    X = data[:, 1:-1]
    y = data[:, -1]

    dt01 = DT.DecisionTree()
    dt01.train(X,y)
          


