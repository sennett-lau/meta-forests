from load_data import load_pacs_training_dataset, load_pacs_testing_dataset, load_vlcs_dataset
from feature_extraction import feature_extract_resnet, feature_extract_decaf6
import random
from models import MetaForests, RandomForest, baseline_random_forest_fit
import numpy as np
import time
import torch
import os
import pickle

def vlcs_load_and_extract_features(
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    feature_dir: str = 'src/feature_extraction',
    force_extract: bool = False
):
    """
    Load and extract features for the VLCS dataset.
    If features already exist on disk, they will be loaded instead of re-extracted.
    
    Args:
        device: Device to use for feature extraction
        feature_dir: Directory to save/load features from
        force_extract: If True, features will be re-extracted even if they exist
    """
    print("================================================")
    print("Loading and extracting features for VLCS dataset...")
    
    # Create feature directory if it doesn't exist
    os.makedirs(feature_dir, exist_ok=True)
    
    vlcs_domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
    training_extracted_features = {}
    testing_extracted_features = {}
    
    # Try to load saved features first
    features_path = os.path.join(feature_dir, 'vlcs_features.pkl')
    if os.path.exists(features_path) and not force_extract:
        print(f"Loading pre-extracted features from {features_path}")
        with open(features_path, 'rb') as f:
            saved_data = pickle.load(f)
            return saved_data['domains'], saved_data['training'], saved_data['testing']
    
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
    
    # Save the extracted features
    print(f"Saving extracted features to {features_path}")
    with open(features_path, 'wb') as f:
        pickle.dump({
            'domains': vlcs_domains,
            'training': training_extracted_features,
            'testing': testing_extracted_features
        }, f)
    
    print("================================================")
    return vlcs_domains, training_extracted_features, testing_extracted_features

def pacs_load_and_extract_features(
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    feature_dir: str = 'src/feature_extraction',
    force_extract: bool = False
):
    """
    Load and extract features for the PACS dataset.
    If features already exist on disk, they will be loaded instead of re-extracted.
    
    Args:
        device: Device to use for feature extraction
        feature_dir: Directory to save/load features from
        force_extract: If True, features will be re-extracted even if they exist
    """
    print("================================================")
    print("Loading and extracting features for PACS dataset...")
    
    # Create feature directory if it doesn't exist
    os.makedirs(feature_dir, exist_ok=True)
    #Photo (1,670 images), Art Painting (2,048 images), Cartoon (2,344 images), and Sketch
    pacs_domains = ['PHOTO', 'ART_PAINTING', 'CARTOON', 'SKETCH']
    training_extracted_features = {}
    testing_extracted_features = {}
    
    # Try to load saved features first
    features_path = os.path.join(feature_dir, 'pacs_features.pkl')
    if os.path.exists(features_path) and not force_extract:
        print(f"Loading pre-extracted features from {features_path}")
        with open(features_path, 'rb') as f:
            saved_data = pickle.load(f)
            return saved_data['domains'], saved_data['training'], saved_data['testing']
    
    # Extract features for each domain
    pacs_training_dataset = load_pacs_training_dataset()
    for i, domain in enumerate(pacs_domains):
        print(f"Extracting training features for '{domain}' domain...")
        pacs_dataset = [item for item in list(pacs_training_dataset) if item["domains"][0] == i]
        features_array, labels_array = feature_extract_resnet(pacs_dataset, device=device)
        training_extracted_features[domain] = (features_array, labels_array)
    
    # Extract features for each domain
    pacs_testing_dataset = load_pacs_testing_dataset()
    for i, domain in enumerate(pacs_domains):
        print(f"Extracting testing features for '{domain}' domain...")
        pacs_dataset = [item for item in list(pacs_testing_dataset) if item["domains"][0] == i]
        features_array, labels_array = feature_extract_resnet(pacs_dataset, device=device)
        testing_extracted_features[domain] = (features_array, labels_array)
    
    # Save the extracted features
    print(f"Saving extracted features to {features_path}")
    with open(features_path, 'wb') as f:
        pickle.dump({
            'domains': pacs_domains,
            'training': training_extracted_features,
            'testing': testing_extracted_features
        }, f)
    
    print("================================================")
    return pacs_domains, training_extracted_features, testing_extracted_features

def meta_forests_on_vlcs(
        epochs: int = 10,
        alpha: float = -1.0,
        beta: float = 1.0,
        epsilon: float = 1e-10,
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
    baseline_rf_model = baseline_random_forest_fit(
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

def meta_forests_on_pacs(
        epochs: int = 10,
        alpha: float = -1.0,
        beta: float = 1.0,
        epsilon: float = 1e-10,
        random_state: int = 42,
        baseline_random_state: int = 52,
        per_random_forest_n_estimators: int = 100,
        per_random_forest_max_depth: int = 5,
        pacs_domains: list[str] = None,
        training_extracted_features: dict = None,
        testing_extracted_features: dict = None
    ):
    # Sample code for PACS dataset loading and feature extraction
    # pacs_dataset = load_pacs_training_dataset()
    # features_array, labels_array = feature_extract_resnet(pacs_dataset)
    # print(features_array.shape)
    # print(labels_array.shape)

    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("================================================")
    print("MetaForests for PACS")

    if pacs_domains is None or training_extracted_features is None or testing_extracted_features is None:
        pacs_domains, training_extracted_features, testing_extracted_features = pacs_load_and_extract_features(device=device)

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
        domains=pacs_domains,
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
    baseline_test_domain = random.choice(pacs_domains)
    baseline_train_domains = [d for d in pacs_domains if d != baseline_test_domain]

    X_baseline_train = np.vstack([training_extracted_features[d][0] for d in baseline_train_domains])
    y_baseline_train = np.hstack([training_extracted_features[d][1] for d in baseline_train_domains])

    X_baseline_test, y_baseline_test = testing_extracted_features[baseline_test_domain]

    print("Training baseline model...")
    print("================================================")
    baseline_rf_model = baseline_random_forest_fit(
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

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    meta_forests_on_vlcs()
    meta_forests_on_pacs()
