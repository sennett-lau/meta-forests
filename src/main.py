from load_data import load_pacs_training_dataset, load_vlcs_dataset
from feature_extraction import feature_extract_resnet, feature_extract_decaf6
import random
from models import MetaForests, random_forest_fit
import numpy as np
import time
import torch

def meta_forests_on_vlcs():
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("================================================")
    print("MetaForests for VLCS")
    vlcs_domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
    training_extracted_features = {}
    testing_extracted_features = {}

    epochs = 20
    random_state = 42
    alpha = -1.0
    beta = 1.0
    epsilon = 1e-6
    random_state = 42
    baseline_random_state = 42
    per_random_forest_n_estimators = 100
    per_random_forest_max_depth = 5

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
    print("Extracting training features for each domain...")
    print("================================================")
    # Extract features for each domain
    for domain in vlcs_domains:
        print(f"Extracting features for '{domain}' domain...")
        vlcs_dataset = load_vlcs_dataset(domain=domain, split='train')
        features_array, labels_array = feature_extract_decaf6(vlcs_dataset, device=device)
        training_extracted_features[domain] = (features_array, labels_array)
    print("================================================")
    print("Extracting testing features for each domain...")
    print("================================================")
    # Extract features for each domain
    for domain in vlcs_domains:
        print(f"Extracting features for '{domain}' domain...")
        vlcs_dataset = load_vlcs_dataset(domain=domain, split='test')
        features_array, labels_array = feature_extract_decaf6(vlcs_dataset)
        testing_extracted_features[domain] = (features_array, labels_array)
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
    baseline_rf_model = random_forest_fit(X_baseline_train, y_baseline_train)
    print("================================================")
    print("Evaluating models...")
    print("================================================")
    baseline_accuracy = baseline_rf_model.score(X_baseline_test, y_baseline_test)
    meta_forests_accuracy = meta_forests.score(X_baseline_test, y_baseline_test)

    print(f"Meta-Forests accuracy on '{baseline_test_domain}' domain: {meta_forests_accuracy:.4f}")
    print(f"Baseline accuracy on '{baseline_test_domain}' domain: {baseline_accuracy:.4f}")
    print(f"Improvement: {meta_forests_accuracy - baseline_accuracy:.4f}")
    print("================================================")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("================================================")

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
