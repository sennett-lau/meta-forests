from load_data import load_pacs_training_dataset, load_vlcs_dataset
from feature_extraction import feature_extract_resnet, feature_extract_decaf6
import random
from models import MetaForests, random_forest_fit
import numpy as np
import time
import torch

def vlcs_load_and_extract_features(
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Load and extract features for the VLCS dataset.
    """
    print("================================================")
    print("Loading and extracting features for VLCS dataset...")
    print("================================================")
    vlcs_domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
    training_extracted_features = {}
    testing_extracted_features = {}
    

    # Extract features for each domain
    for domain in vlcs_domains:
        print(f"Extracting training features for '{domain}' domain...")
        vlcs_dataset = load_vlcs_dataset(domain=domain, split='train')
        features_array, labels_array = feature_extract_decaf6(vlcs_dataset, device=device)
        training_extracted_features[domain] = (features_array, labels_array)
    
    # Extract features for each domain
    for domain in vlcs_domains:
        print(f"Extracting testing features for '{domain}' domain...")
        vlcs_dataset = load_vlcs_dataset(domain=domain, split='test')
        features_array, labels_array = feature_extract_decaf6(vlcs_dataset, device=device)
        testing_extracted_features[domain] = (features_array, labels_array)
    
    return vlcs_domains, training_extracted_features, testing_extracted_features

def meta_forests_on_vlcs(
        epochs: int = 20,
        alpha: float = -1.0,
        beta: float = 1.0,
        epsilon: float = 1e-6,
        random_state: int = 42,
        baseline_random_state: int = 52,
        per_random_forest_n_estimators: int = 100,
        per_random_forest_max_depth: int = 5,
        vlcs_domains: list[str] = None,
        training_extracted_features: dict = None,
        testing_extracted_features: dict = None
    ):
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("================================================")
    print("MetaForests for VLCS")

    if vlcs_domains is None or training_extracted_features is None or testing_extracted_features is None:
        vlcs_domains, training_extracted_features, testing_extracted_features = vlcs_load_and_extract_features(device=device)

    print("================================================")
    print("Hyperparameters:")
    print(f"Epochs: {epochs}")
    print(f"Alpha: {alpha}")
    print(f"Beta: {beta}")
    print(f"Epsilon: {epsilon}")
    print(f"Random state: {random_state}")
    print(f"Per random forest n estimators: {per_random_forest_n_estimators}")
    print(f"Per random forest max depth: {per_random_forest_max_depth}")
    print("================================================")
    print("Training MetaForests model...")
    print("================================================")
    # Initialize and train the MetaForests model
    meta_forests = MetaForests(
        domains=vlcs_domains,
        extracted_features=training_extracted_features,
        epochs=epochs,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        random_state=random_state,
        per_random_forest_n_estimators=per_random_forest_n_estimators,
        per_random_forest_max_depth=per_random_forest_max_depth
    )
    meta_forests.train()
    print("================================================")
    print("Configuring baseline dataset...")
    print("================================================")
    # Randomly select a test domain and prepare baseline model
    random.seed(baseline_random_state)
    baseline_test_domain = random.choice(vlcs_domains)
    baseline_train_domains = [d for d in vlcs_domains if d != baseline_test_domain]

    X_baseline_train = np.vstack([training_extracted_features[d][0] for d in baseline_train_domains])
    y_baseline_train = np.hstack([training_extracted_features[d][1] for d in baseline_train_domains])

    X_baseline_test, y_baseline_test = testing_extracted_features[baseline_test_domain]

    print("Training baseline model...")
    print("================================================")
    baseline_rf_model = random_forest_fit(
        X_baseline_train,
        y_baseline_train, 
        n_estimators=per_random_forest_n_estimators,
        max_depth=per_random_forest_max_depth,
        random_state=baseline_random_state
    )
    print("================================================")
    print("Evaluating models...")
    print("================================================")
    baseline_accuracy = baseline_rf_model.score(X_baseline_test, y_baseline_test)
    meta_forests_accuracy = meta_forests.score(X_baseline_test, y_baseline_test)

    print(f"Meta-Forests accuracy on '{baseline_test_domain}' domain: {meta_forests_accuracy:.4f}")
    print(f"Baseline accuracy on '{baseline_test_domain}' domain: {baseline_accuracy:.4f}")
    improvement = meta_forests_accuracy - baseline_accuracy
    print(f"Improvement: {improvement:.4f}")
    print("================================================")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("================================================")

    return meta_forests_accuracy, baseline_accuracy, improvement

def meta_forests_on_pacs():
    # Sample code for PACS dataset loading and feature extraction
    # pacs_dataset = load_pacs_training_dataset()
    # features_array, labels_array = feature_extract_resnet(pacs_dataset)
    # print(features_array.shape)
    # print(labels_array.shape)
    pass

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    meta_forests_on_vlcs()
