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

class MetaForests:
    def __init__(
            self,
            domains: list,
            extracted_features: dict,
            epochs: int = 20,
            alpha: float = -1.0,
            beta: float = 1.0,
            epsilon: float = 1e-10,
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
        self.random_states = [random.randint(0, 1000000) for _ in range(self.epochs * (len(domains) - 1))]
        self.per_random_forest_n_estimators = per_random_forest_n_estimators
        self.per_random_forest_max_depth = per_random_forest_max_depth
        
        # Store all forests and weights per iteration
        self.all_iterations_forests = []
        self.all_iterations_weights = []
        self.all_iterations_mmds = []
        self.init_weight = 1.0 / (len(domains) - 2)

        for each_epoch in range(epochs):
            each_forest_weights = []
            for each_forest in range(len(domains) - 2):
                each_forest_weights.append(self.init_weight)
            self.all_iterations_forests.append([])
            self.all_iterations_weights.append(each_forest_weights)
            self.all_iterations_mmds.append([])
            
    def train(self):
        """
        Trains the meta-forests using meta-learning according to the algorithm.
        Please note that we are using i and j from 0, but the algorithm in the paper uses 1.
        """
        normalized_weights = []
        randomIndex = 0
        for i in range(self.epochs):
            normalized_weights.append([])
            random.seed(self.random_states[randomIndex])
            np.random.seed(self.random_states[randomIndex])
            
            # Randomly select 1 domain as D_meta_test
            meta_test_domain = random.choice(self.domains)

            # Randomly select num_of_train_domains domains as D_meta_train while ensuring no duplicates
            remaining_domains = [d for d in self.domains if d != meta_test_domain]
            num_of_train_domains = len(self.domains) - 2
            meta_train_domains = random.sample(remaining_domains, num_of_train_domains)
            
            # For each domain j in meta_train (M-2 domains)
            for j, domain in enumerate(meta_train_domains):
                # Train random forest model on single domain with previous weights
                X_train, y_train = self.extracted_features[domain]
                num_classes = len(np.unique(y_train))  # Get C (number of classes)
                
                rf_model = random_forest_fit(
                    X_train,
                    y_train,
                    self.per_random_forest_n_estimators,
                    self.per_random_forest_max_depth,
                    self.random_states[randomIndex]
                )
                randomIndex += 1
                
                # Calculate W_accuracy
                X_test, y_test = self.extracted_features[meta_test_domain]
                score = rf_model.score(X_test, y_test)
                W_accuracy = self.compute_w_accuracy(score, num_classes)

                # Calculate W_mmd
                mmd_ij = self.compute_mmd(X_train, X_test, kernel='rbf')
                self.all_iterations_mmds[i].append(mmd_ij)
                W_mmd = self.compute_w_mmd(mmd_ij, i, j)

                # Calculate W_ij
                W_ij = self.compute_w_ij(i, j, W_accuracy, W_mmd)
                self.all_iterations_weights[i][j] = W_ij
                
                # Store the model and metrics
                forest_info = {
                    'model': rf_model,
                    'domain': domain,
                }

                self.all_iterations_forests[i].append(forest_info)

                print(f"Iteration {i+1}/{self.epochs} | Domain {j+1}/{len(meta_train_domains)} | "
                      f"Meta-test: {meta_test_domain}, Meta-train: {domain}, "
                      f"MMD: {mmd_ij:.4f}, W_mmd: {W_mmd:.4f}, "
                      f"Accuracy: {W_accuracy:.4f}, Weight: {W_ij:.6f}")
            
            # Normalize weights for this iteration
            total_weight = sum(self.all_iterations_weights[i])
            normalized_weights[i] = [w / total_weight for w in self.all_iterations_weights[i]]
        
        # Format the output to match algorithm's expected format
        self.meta_forests = []
        self.meta_weights_normalized = []
        
        # Store the latest iteration's models and weights
        for forest, weight in zip(self.all_iterations_forests[-1], normalized_weights[-1]):
            self.meta_forests.append(forest)
            self.meta_weights_normalized.append(weight)

    def predict(self, test_features: np.ndarray) -> np.ndarray:
        """
        Predicts using the weighted ensemble of meta-forests from the final iteration.
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
    
    def compute_w_accuracy(self, score, num_classes):
        accuracy_term = np.exp(score)  # This approximates e^(∑ᵢ I(yₙ=ŷₙ)/S)
        baseline_term = np.exp(1/num_classes)
        return max(self.epsilon, accuracy_term - baseline_term)

    def compute_mmd(self, X, Y, kernel='rbf', gamma=None):
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
    
    def compute_w_mmd(self, mmd_ij, i, j):
        if i == 0 or j == 0:  # Handle edge cases
            return mmd_ij

        sum_mmds = 0
        for prev_i in range(i):
            for prev_j in range(j):
                sum_mmds += self.all_iterations_mmds[prev_i][prev_j]
        # Fix: Use (i-1)*(j-1) in the denominator
        return mmd_ij - sum_mmds / ((i) * (j))

    def compute_w_ij(self, i, j, w_accuracy, w_mmd):
        prev_i = max(0, i - 1)
        prev_w = self.all_iterations_weights[prev_i][j]
        mmd_factor = np.exp(self.alpha * w_mmd)
        # Fix: Use log^(βWaccuracy) instead of log(β*Waccuracy)
        accuracy_factor = np.power(np.log(w_accuracy + self.epsilon), self.beta)
        return prev_w * mmd_factor * accuracy_factor
