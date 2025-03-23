from load_data import load_pacs_training_dataset, load_vlcs_dataset
from feature_extraction import feature_extract_resnet, feature_extract_decaf6
import random
from models import vlcs_random_forest
import numpy as np

def main():
    vlcs_domains = ['CALTECH', 'LABELME', 'PASCAL', 'SUN']
    extracted_features = {}
    for domain in vlcs_domains:
        vlcs_dataset = load_vlcs_dataset(domain=domain, split='train')
        features_array, labels_array = feature_extract_decaf6(vlcs_dataset)
        extracted_features[domain] = (features_array, labels_array)

    # randomly select a domain
    test_domain = random.choice(vlcs_domains)
    train_domains = [domain for domain in vlcs_domains if domain != test_domain]

    # Concatenate training domain features and labels clearly
    X_train = np.vstack([extracted_features[domain][0] for domain in train_domains])
    y_train = np.hstack([extracted_features[domain][1] for domain in train_domains])

    # Clearly separate test domain features and labels
    X_test, y_test = extracted_features[test_domain]

    # Quick check on data shapes
    print(f"Selected test domain: {test_domain}")
    print(f"Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

    # Train random forest model on VLCS dataset clearly
    rf_model = vlcs_random_forest(X_train, y_train)

    # Evaluate clearly on test data
    test_accuracy = rf_model.score(X_test, y_test)
    print(f"Test accuracy on '{test_domain}' domain: {test_accuracy:.4f}")

    
    # Sample code for PACS dataset loading and feature extraction
    # pacs_dataset = load_pacs_training_dataset()
    # features_array, labels_array = feature_extract_resnet(pacs_dataset)
    # print(features_array.shape)
    # print(labels_array.shape)

if __name__ == "__main__":
    main()
