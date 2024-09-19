

!pip install matplotlib

!pip install datasets transformers numpy scikit-learn

!pip install transformers[torch]

!pip install datasets



# [Author: Sai Kiran Gandluri(02144549)]
# [Date: 03-22-2024]

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

def tokenize_function(examples, tokenizer):
   
    return tokenizer(examples['premise'], examples['hypothesis'], padding="max_length", truncation=True, max_length=512)

def evaluate_model(trainer, dataset):
    predictions = trainer.predict(dataset).predictions
    predictions = np.argmax(predictions, axis=1)
    labels = dataset['label']
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='macro')
    return precision, recall, f1

def main():
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    # Load datasets directly from CSV and Mention Correct path name 
    train_dataset = load_dataset('csv', data_files='/path to/SDOH-NLI-main/sdoh_nli_train.csv')['train']
    val_dataset = load_dataset('csv', data_files='/path to/SDOH-NLI-main/sdoh_nli_validation.csv')['train']
    test_dataset = load_dataset('csv', data_files='/path to/SDOH-NLI-main/sdoh_nli_test.csv')['train']
    twitter_test_dataset = load_dataset('csv', data_files='/path to/tweet_test.csv')['train']

    # Tokenize the datasets
    train_dataset = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    twitter_test_dataset = twitter_test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)

    # bert-base-uncased for analysis
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=9,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=6,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
    )

    # fine-tuning the models on training data and evaluate it on validation data
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train()

    metrics = []

    # evaluating the trained model on validation and test data, and on twitter test data
    for dataset, name in zip([val_dataset, test_dataset, twitter_test_dataset], ["Validation", "Test", "Twitter Test"]):
        precision, recall, f1 = evaluate_model(trainer, dataset)
        metrics.append((precision, recall, f1))
        print(f"Test Results - Precision: {precision:.4f}, "
         f"Recall: {recall:.4f}, "
         f"F1: {f1:.4f}")
    # Plotting the metrics
    epochs = ["Validation", "Test", "Twitter Test"]
    precisions = [m[0] for m in metrics]
    recalls = [m[1] for m in metrics]
    f1_scores = [m[2] for m in metrics]

    plt.plot(epochs, precisions, label='Precision', marker='o')
    plt.plot(epochs, recalls, label='Recall', marker='x')
    plt.plot(epochs, f1_scores, label='F1 Score', marker='s')

    plt.title('Model Performance Metrics Across Datasets')
    plt.ylabel('Scores')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()

