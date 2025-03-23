from load_data import load_pacs_training_dataset, load_vlcs_dataset
from feature_extraction import feature_extract_resnet, feature_extract_decaf6
import random
from models import vlcs_random_forest, compute_mmd
import numpy as np

def main():
    vlcs_domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
    extracted_features = {}
    
    # Extract features clearly for each domain
    for domain in vlcs_domains:
        vlcs_dataset = load_vlcs_dataset(domain=domain, split='train')
        features_array, labels_array = feature_extract_decaf6(vlcs_dataset)
        extracted_features[domain] = (features_array, labels_array)

    # Meta-learning loop: clearly iterate N times
    N = 20  # Clearly set according to the paper recommendation
    meta_forests = []

    for iteration in range(N):
        # Clearly select meta-test and meta-train subsets from train_domains
        meta_test_domain = random.choice(vlcs_domains)
        meta_train_domains = [d for d in vlcs_domains if d != meta_test_domain]

        # Clearly prepare meta-train data
        X_meta_train = np.vstack([extracted_features[d][0] for d in meta_train_domains])
        y_meta_train = np.hstack([extracted_features[d][1] for d in meta_train_domains])

        # Clearly prepare meta-test data
        X_meta_test, y_meta_test = extracted_features[meta_test_domain]

        # Train random forest model clearly
        rf_model = vlcs_random_forest(X_meta_train, y_meta_train)

        # Evaluate accuracy clearly on meta-test set
        accuracy = rf_model.score(X_meta_test, y_meta_test)
        
        mmd_distance = compute_mmd(X_meta_train, X_meta_test, kernel='rbf', gamma=None)
        
        # Clearly store model and meta information
        meta_forests.append({
            'model': rf_model,
            'accuracy': accuracy,
            'mmd_distance': mmd_distance,
            'meta_train_domains': meta_train_domains,
            'meta_test_domain': meta_test_domain
        })

        print(f"Iteration {iteration+1}/{N} | Meta-test domain: {meta_test_domain}, Accuracy: {accuracy:.4f}")

    
    # Sample code for PACS dataset loading and feature extraction
    # pacs_dataset = load_pacs_training_dataset()
    # features_array, labels_array = feature_extract_resnet(pacs_dataset)
    # print(features_array.shape)
    # print(labels_array.shape)

if __name__ == "__main__":
    main()
