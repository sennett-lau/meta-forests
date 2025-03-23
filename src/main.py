from load_data import load_pacs_training_dataset, load_vlcs_dataset
from feature_extraction import feature_extract_resnet, feature_extract_decaf6

def main():
    # Example: Load the SUN domain VLCS data
    vlcs_sun_dataset = load_vlcs_dataset(domain='SUN', split='full')

    # Checking the loader
    for batch in vlcs_sun_dataset:
        images, labels = batch['images'], batch['labels']
        print(images.shape, labels.shape)
        break

    features_array, labels_array = feature_extract_decaf6(vlcs_sun_dataset)
    print(features_array.shape)
    print(labels_array.shape)

    pacs_dataset = load_pacs_training_dataset()
    features_array, labels_array = feature_extract_resnet(pacs_dataset)
    print(features_array.shape)
    print(labels_array.shape)

if __name__ == "__main__":
    main()
