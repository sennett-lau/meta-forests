from load_data import load_pacs_training_dataset, load_pacs_testing_dataset, load_vlcs_dataset
from feature_extraction import feature_extract_resnet, feature_extract_decaf6
from models import MetaForests, baseline_random_forest_fit
import numpy as np
import time
import torch
import os
import pickle
import pandas as pd

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
    
    vlcs_domains = ['PASCAL', 'CALTECH', 'LABELME', 'SUN']
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
        alpha: float = -0.5,
        beta: float = 0.5,
        epsilon: float = 1e-10,
        random_state: int = 42,
        baseline_random_state: int = 52,
        per_random_forest_n_estimators: int = 100,
        per_random_forest_max_depth: int = 5,
        vlcs_domains: list[str] = None,
        vlcs_target_domain: str = None,
        training_extracted_features: dict = None,
        testing_extracted_features: dict = None,
        mmd_kernel: str = 'rbf',
    ):
    start_time = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("================================================")
    print("MetaForests for VLCS")

    if vlcs_domains is None or training_extracted_features is None or testing_extracted_features is None:
        vlcs_domains, training_extracted_features, testing_extracted_features = vlcs_load_and_extract_features(device=device)
    
    if vlcs_target_domain is None:
        vlcs_target_domain = vlcs_domains[0]
    
    vlcs_source_domains = [d for d in vlcs_domains if d != vlcs_target_domain]

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
        target_domain=vlcs_target_domain,
        extracted_features=training_extracted_features,
        epochs=epochs,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        random_state=random_state,
        per_random_forest_n_estimators=per_random_forest_n_estimators,
        per_random_forest_max_depth=per_random_forest_max_depth,
        mmd_kernel=mmd_kernel
    )
    meta_forests.train()
    print("================================================")
    print("Configuring baseline dataset...")
    print("================================================")

    X_baseline_train = np.vstack([training_extracted_features[d][0] for d in vlcs_source_domains])
    y_baseline_train = np.hstack([training_extracted_features[d][1] for d in vlcs_source_domains])

    X_baseline_test, y_baseline_test = training_extracted_features[vlcs_target_domain]

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
    rf_baseline_accuracy = baseline_rf_model.score(X_baseline_test, y_baseline_test)
    meta_forests_accuracy = meta_forests.score(X_baseline_test, y_baseline_test)

    print(f"Meta-Forests accuracy on '{vlcs_target_domain}' domain: {meta_forests_accuracy:.4f}")
    print(f"Baseline accuracy on '{vlcs_target_domain}' domain: {rf_baseline_accuracy:.4f}")
    improvement = meta_forests_accuracy - rf_baseline_accuracy
    print(f"Improvement: {improvement:.4f}")
    print("================================================")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("================================================")

    return meta_forests_accuracy, rf_baseline_accuracy, improvement

def meta_forests_on_pacs(
        epochs: int = 10,
        alpha: float = -0.5,
        beta: float = 0.5,
        epsilon: float = 1e-10,
        random_state: int = 42,
        baseline_random_state: int = 52,
        per_random_forest_n_estimators: int = 100,
        per_random_forest_max_depth: int = 5,
        pacs_domains: list[str] = None,
        pacs_target_domain: str = None,
        training_extracted_features: dict = None,
        testing_extracted_features: dict = None,
        mmd_kernel: str = 'rbf',
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

    if pacs_target_domain is None:
        pacs_target_domain = pacs_domains[0]
    
    pacs_source_domains = [d for d in pacs_domains if d != pacs_target_domain]


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
        target_domain=pacs_target_domain,
        extracted_features=training_extracted_features,
        epochs=epochs,
        alpha=alpha,
        beta=beta,
        epsilon=epsilon,
        random_state=random_state,
        per_random_forest_n_estimators=per_random_forest_n_estimators,
        per_random_forest_max_depth=per_random_forest_max_depth,
        mmd_kernel=mmd_kernel
    )
    meta_forests.train()
    print("================================================")
    print("Configuring baseline dataset...")
    print("================================================")

    X_baseline_train = np.vstack([training_extracted_features[d][0] for d in pacs_source_domains])
    y_baseline_train = np.hstack([training_extracted_features[d][1] for d in pacs_source_domains])

    X_baseline_test, y_baseline_test = testing_extracted_features[pacs_target_domain]

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
    rf_baseline_accuracy = baseline_rf_model.score(X_baseline_test, y_baseline_test)
    meta_forests_accuracy = meta_forests.score(X_baseline_test, y_baseline_test)

    print(f"Meta-Forests accuracy on '{pacs_target_domain}' domain: {meta_forests_accuracy:.4f}")
    print(f"Baseline accuracy on '{pacs_target_domain}' domain: {rf_baseline_accuracy:.4f}")
    improvement = meta_forests_accuracy - rf_baseline_accuracy
    print(f"Improvement: {improvement:.4f}")
    print("================================================")
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("================================================")

    return meta_forests_accuracy, rf_baseline_accuracy, improvement

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Using device: {device}")

    vlcs_domains, vlcs_training_extracted_features, vlcs_testing_extracted_features = vlcs_load_and_extract_features(device=device)
    pacs_domains, pacs_training_extracted_features, pacs_testing_extracted_features = pacs_load_and_extract_features(device=device)
    
    results = []

    for domain in vlcs_domains:
        meta_forests_accuracy, rf_baseline_accuracy, improvement = meta_forests_on_vlcs(
            vlcs_domains=vlcs_domains,
            vlcs_target_domain=domain,
            training_extracted_features=vlcs_training_extracted_features,
            testing_extracted_features=vlcs_testing_extracted_features
        )
        results.append({
            'dataset': 'VLCS',
            'domain': domain,
            'meta_forests_accuracy': meta_forests_accuracy,
            'rf_baseline_accuracy': rf_baseline_accuracy,
            'improvement': improvement
        })

        
    for domain in pacs_domains:
        meta_forests_accuracy, rf_baseline_accuracy, improvement = meta_forests_on_pacs(
            pacs_domains=pacs_domains,
            pacs_target_domain=domain,
            training_extracted_features=pacs_training_extracted_features,
            testing_extracted_features=pacs_testing_extracted_features
        )
        results.append({
            'dataset': 'PACS',
            'domain': domain,
            'meta_forests_accuracy': meta_forests_accuracy,
            'rf_baseline_accuracy': rf_baseline_accuracy,
            'improvement': improvement
        })

    results_df = pd.DataFrame(results)
    
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(curr_dir, 'results', 'accuracy_results.csv')
    results_df.to_csv(save_path, index=False)
