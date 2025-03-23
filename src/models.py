import numpy as np
import random
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.ensemble import RandomForestClassifier

def random_forest_fit(X_train, y_train, n_estimators=100, max_depth=5, random_state=None):
    """
    Trains a Random Forest model on VLCS dataset features clearly.

    Parameters:
    - X_train (np.ndarray): Training features clearly.
    - y_train (np.ndarray): Training labels clearly.
    - n_estimators (int): Number of trees clearly.
    - max_depth (int): Max depth of trees clearly.
    - random_state (int): Random state for reproducibility clearly.

    Returns:
    - Trained RandomForestClassifier model clearly.
    """
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    rf.fit(X_train, y_train)
    return rf

def compute_mmd(X, Y, kernel='rbf', gamma=None):
    """
    Clearly computes the Maximum Mean Discrepancy (MMD) between two datasets X and Y.

    Parameters:
        X (np.ndarray): Samples from the first distribution (meta-train).
        Y (np.ndarray): Samples from the second distribution (meta-test).
        kernel (str): Kernel type (default: 'rbf').
        gamma (float): Kernel coefficient for 'rbf'. If None, uses 1/n_features.

    Returns:
        float: MMD distance between the two distributions.
    """
    XX = pairwise_kernels(X, X, metric=kernel, gamma=gamma)
    YY = pairwise_kernels(Y, Y, metric=kernel, gamma=gamma)
    XY = pairwise_kernels(X, Y, metric=kernel, gamma=gamma)

    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

class MetaForests:
    def __init__(
            self,
            domains: list,
            extracted_features: dict,
            epochs: int = 20,
            alpha: float = -1.0,
            beta: float = 1.0,
            epsilon: float = 1e-6,
            random_state: int = 42,
            per_random_forest_n_estimators: int = 100,
            per_random_forest_max_depth: int = 5
        ):
        """
        Initializes the MetaForests class.

        Parameters:
        - domains (list): List of domain names.
        - extracted_features (dict): Dictionary containing features and labels for each domain. Each value is a tuple (features_array, labels_array).
        - epochs (int): Number of meta-learning iterations.
        - alpha (float): Parameter controlling the impact of MMD on weight updates.
        - beta (float): Parameter controlling the impact of accuracy on weight updates.
        - epsilon (float): Small constant to avoid numerical issues.
        - random_state (int): Seed for reproducibility.
        - per_random_forest_n_estimators (int): Number of trees in each random forest.
        - per_random_forest_max_depth (int): Max depth of trees in each random forest.
        """
        self.domains = domains
        self.extracted_features = extracted_features
        self.epochs = epochs
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.random_state = random_state
        self.meta_forests = []
        self.meta_weights_normalized = []
        random.seed(self.random_state)
        self.random_states = [random.randint(0, 1000000) for _ in range(self.epochs)]
        self.per_random_forest_n_estimators = per_random_forest_n_estimators
        self.per_random_forest_max_depth = per_random_forest_max_depth

    def train(self):
        """
        Trains the meta-forests using meta-learning.
        """
        meta_weights = []

        for epoch in range(self.epochs):
            random.seed(self.random_states[epoch])
            np.random.seed(self.random_states[epoch])

            meta_test_domain = self.domains[epoch % len(self.domains)]
            meta_train_domains = [d for d in self.domains if d != meta_test_domain]

            X_meta_train = np.vstack([self.extracted_features[d][0] for d in meta_train_domains])
            y_meta_train = np.hstack([self.extracted_features[d][1] for d in meta_train_domains])

            X_meta_test, y_meta_test = self.extracted_features[meta_test_domain]

            rf_model = random_forest_fit(X_meta_train, y_meta_train, self.per_random_forest_n_estimators, self.per_random_forest_max_depth, self.random_states[epoch])

            accuracy = rf_model.score(X_meta_test, y_meta_test)
            mmd_distance = compute_mmd(X_meta_train, X_meta_test, kernel='rbf', gamma=None)

            W_mmd = np.exp(self.alpha * mmd_distance)
            W_accuracy = np.log(self.beta * accuracy + self.epsilon)
            W_current = max(W_mmd * W_accuracy, self.epsilon)

            meta_weights.append(W_current)
            total_weight = sum(meta_weights)
            self.meta_weights_normalized = [w / total_weight for w in meta_weights]

            self.meta_forests.append({
                'model': rf_model,
                'weight': W_current,
                'accuracy': accuracy,
                'mmd_distance': mmd_distance
            })

            print(f"Epoch {epoch+1}/{self.epochs} | Meta-test domain: {meta_test_domain}, Accuracy: {accuracy:.4f}")

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predicts the labels for the given test features using the trained meta-forests.

        Parameters:
        - test_features (np.ndarray): Array of test features.

        Returns:
        - np.ndarray: Predicted labels.
        """
        predictions_weighted = np.zeros((test_features.shape[0], len(np.unique(self.extracted_features[self.domains[0]][1]))))
        
        for model_info, normalized_weight in zip(self.meta_forests, self.meta_weights_normalized):
            rf_model = model_info['model']
            preds_proba = rf_model.predict_proba(test_features)
            predictions_weighted += normalized_weight * preds_proba

        final_predictions = np.argmax(predictions_weighted, axis=1)
        return final_predictions

    def score(self, test_features: np.ndarray, test_labels: np.ndarray) -> float:
        """
        Computes the accuracy score of the MetaForests on the given test data.

        Parameters:
        - test_features (np.ndarray): Array of test features.
        - test_labels (np.ndarray): True labels for the test data.

        Returns:
        - float: Accuracy score.
        """
        predictions = self.predict(test_features)
        accuracy = np.mean(predictions == test_labels)
        return accuracy
