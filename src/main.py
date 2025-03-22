from load_data import load_pacs_training_dataset
from feature_extraction import feature_extract_resnet

def main():
    pacs_dataset = load_pacs_training_dataset()
    features_array, labels_array = feature_extract_resnet(pacs_dataset)
    print(features_array.shape)
    print(labels_array.shape)

if __name__ == "__main__":
    main()
