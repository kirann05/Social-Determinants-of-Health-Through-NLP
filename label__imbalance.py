


from datasets import load_dataset
import pandas as pd

def load_and_summarize_datasets(train_path, val_path, test_path, twitter_test_path):
    # Load datasets
    train_dataset = load_dataset('csv', data_files=train_path)['train']
    val_dataset = load_dataset('csv', data_files=val_path)['train']
    test_dataset = load_dataset('csv', data_files=test_path)['train']
    twitter_test_dataset = load_dataset('csv', data_files=twitter_test_path)['train']

    datasets = {
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
        "twitter_test": twitter_test_dataset
    }

    # Summarize datasets
    for name, dataset in datasets.items():
        print(f"Summary for {name} dataset:")
        df = pd.DataFrame(dataset)
        print(f"Total examples: {len(df)}")
        if 'label' in df.columns:
            print("Label distribution:")
            print(df['label'].value_counts(normalize=True).to_string())
        else:
            print("No 'label' column found for label distribution.")
        print("-" * 50)

# Define your dataset paths
train_path = '/content/drive/MyDrive/SDOH-NLI-main/sdoh_nli_train.csv'
val_path = '/content/drive/MyDrive/SDOH-NLI-main/sdoh_nli_validation.csv'
test_path = '/content/drive/MyDrive/SDOH-NLI-main/sdoh_nli_test.csv'
twitter_test_path = '/content/drive/MyDrive/tweet_test.csv'

# Call the function with the dataset paths
load_and_summarize_datasets(train_path, val_path, test_path, twitter_test_path)
