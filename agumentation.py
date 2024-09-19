

!pip install datasets

!pip install transformers

!pip install datasets transformers numpy scikit-learn

pip install transformers[torch]

!pip install pandas



!pip install nltk

import nltk
nltk.download('wordnet')  # WordNet is a lexical database for the English language in NLTK.
nltk.download('stopwords')  # Stopwords are a list of high frequency words like the, to and also that we sometimes want to filter out of a document before further processing.
nltk.download('averaged_perceptron_tagger')  # Tagger that assigns tags to words (like noun, verb, etc.)

import pandas as pd
from nltk.corpus import stopwords, wordnet
import nltk
import random

# Setup NLTK
nltk.download('wordnet')
nltk.download('stopwords')

# Function to get synonyms
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace("_", " ").replace("-", " ").lower()
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

# Function for synonym replacement
def synonym_replacement(sentence, n):
    words = sentence.split()
    new_words = words[:]  # Make a copy of the words list
    random_word_list = list(set([word for word in words if word.lower() not in stopwords.words('english')]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        syns = get_synonyms(random_word)
        if len(syns) > 0:
            synonym = random.choice(syns)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # Only replace up to n words
            break

    return ' '.join(new_words)

# Paths to your dataset CSV files
file_paths = {
    'train': '/path/sdoh_nli_train.csv',
    'validation': '/path to/sdoh_nli_validation.csv',
    'test': '/path to/SDOH-NLI-main/sdoh_nli_test.csv'
}

# Apply data augmentation
for dataset, file_path in file_paths.items():
    df = pd.read_csv(file_path)
    # Apply synonym replacement augmentation to 'premise' and 'hypothesis'
    df['augmented_premise'] = df['premise'].apply(lambda x: synonym_replacement(x, 3))
    df['augmented_hypothesis'] = df['hypothesis'].apply(lambda x: synonym_replacement(x, 3))
    # Define the full path where you want to save the file
    augmented_file_path = f'/path to/Augmented_Datasets/augmented_{dataset}.csv'
    df.to_csv(augmented_file_path, index=False)
    print(f"Augmented {dataset} dataset saved to {augmented_file_path}")

    # Print 2 examples of augmented data
    print(f"Showing 2 examples of augmented data for {dataset}:")
    print(df[['augmented_premise', 'augmented_hypothesis']].head(2))
    print(f"Rest of the data saved to {augmented_file_path}\n")

