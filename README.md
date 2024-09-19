# Social-Determinants-of-Health-Through-NLP
Here's a detailed description you can use for your GitHub repository on the *Social Determinants of Health (SDOH) Extraction* project:

---

# **TweetNLP and Social Determinants of Health (SDOH) Extraction**

This project focuses on leveraging Natural Language Processing (NLP) techniques to extract Social Determinants of Health (SDOH) from text data, specifically from clinical notes and social media (tweets). The goal is to accurately categorize and analyze various social factors such as socioeconomic status, education level, physical environment, and behaviors like smoking or dietary habits that influence individuals' health outcomes.

## **Project Overview**
The project implements and critically evaluates a BERT-based NLP model inspired by the research paper *"SDOH-NLI: A Dataset for Inferring Social Determinants of Health from Clinical Notes"* by Lelkes et al. It involves exploring, applying, and enhancing the model using precision, recall, and F1 score metrics to improve its ability to extract SDOH from clinical and social media data.

### **Key Features:**
- **Data Sources**: The project uses the *SDOH-NLI* dataset, which contains clinical notes, and an additional dataset of tweets where the premise is a real-world tweet and the hypothesis determines the truth value based on the premise. 
  - Dataset Link: [SDOH-NLI Dataset](https://github.com/google-research-datasets/SDOH-NLI/tree/main)
  
- **Model**: The core of this project is a BERT-based sequence classification model, fine-tuned to identify various SDOH categories.
  
- **Model Evaluation**: The model’s performance is evaluated using precision, recall, and F1-score metrics on both validation and test datasets. Special attention is given to label imbalances and performance on real-world tweets.

## **Methodology**
### **1. Data Preprocessing**
The datasets consist of clinical notes and tweets labeled with various social determinants. The text data is preprocessed by:
- Tokenizing the text using the Hugging Face `AutoTokenizer` for the BERT model.
- Converting input text into token representations suitable for the model.
- Handling missing data and removing irrelevant features like URLs and symbols from the text.

### **2. Model Fine-Tuning**
- A BERT model (`bert-base`) is fine-tuned using the Hugging Face `Trainer` API. 
- The model is trained on clinical notes to classify SDOH categories.
- The fine-tuning process involves hyperparameter optimization such as adjusting the learning rate, batch size, and epochs to balance between overfitting and underfitting.

### **3. Model Evaluation**
- Precision, Recall, and F1-score are used to evaluate model performance across validation and test datasets.
- Results show:
  - **Validation Set**: F1 score of 91.35%, demonstrating strong performance.
  - **Test Set**: Precision of 92.92% and F1 score of 86.72%, revealing slight performance drops in unseen data.
  - **Twitter Data**: The model shows moderate performance with an F1 score of 66.67% due to the complexity of Twitter’s informal language structure.

### **4. Alternative Approaches**
To improve the model, several strategies were proposed and tested:
- **Data Augmentation**: Synonym substitution was applied to the training dataset to address label imbalance and enhance model generalization. This led to a significant boost in precision, recall, and F1 score.
- **Exploratory Data Analysis (EDA)**: EDA was conducted on the SDOH-NLI dataset to identify label imbalances and potential biases. The imbalance between labels "False" and "True" was adjusted to improve model fairness.
  
### **5. Ethical Considerations**
The project carefully examined the ethical implications of using NLP models for extracting SDOH:
- **Data Privacy**: Ensured that all personal health information (PHI) was anonymized to protect patient confidentiality.
- **Fairness**: Addressed biases in the dataset by auditing diversity representation and ensuring model predictions are fair across different demographic groups.
- **Transparency**: Provided detailed documentation of the model training process to enhance transparency and reproducibility.

## **Results**
- **Precision, Recall, and F1-score Analysis**: The model’s overall performance is robust on the validation and test datasets, although it shows some limitations when applied to real-world Twitter data due to the informal nature of tweets.
- **Improvements via Data Augmentation**: By applying synonym substitution and balancing the dataset, the model’s F1 score improved by approximately 5%, showcasing the effectiveness of these techniques in enhancing NLP models in healthcare.
  
## **Challenges**
- **Label Imbalance**: A significant portion of the dataset was skewed towards "False" labels, making it difficult for the model to generalize to "True" categories.
- **Real-World Application on Twitter**: The informal and slang-heavy nature of tweets posed challenges for accurate SDOH classification. Further fine-tuning and dataset expansion would be necessary to improve performance in this domain.

## **Future Work**
- **Improved Tweet Handling**: Future improvements include developing more sophisticated text preprocessing techniques to handle the noisy, unstructured language found in tweets.
- **Adversarial Training**: Implementing adversarial training to better handle evolving narratives and misinformation on social media.
- **Real-Time SDOH Monitoring**: Deploying the model in a real-time application that can continuously analyze social media posts for social determinants of health.

## **Conclusion**
This project highlights the potential of NLP models in identifying Social Determinants of Health from clinical notes and social media. The model shows promising results, though further work is needed to improve its robustness, especially in handling informal social media text. Ethical considerations such as data privacy, fairness, and transparency are paramount in applying NLP techniques in healthcare, ensuring responsible usage and equitable outcomes.

## **Repository Structure**
- **/data/**: Contains processed datasets (not included in the repo for privacy reasons).
- **/code.py**: Python implementation of the BERT-based model for SDOH extraction.
- **/report/**: A 2-page report discussing the model performance, alternative approaches, and ethical reflections.

## **How to Run**
1. Clone the repository:
   ```
   git clone https://github.com/your-repo-link
   ```
2. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the model:
   ```
   python code.py
   ```

## **References**
- Lelkes, D., et al. *SDOH-NLI: A Dataset for Inferring Social Determinants of Health from Clinical Notes*. EMNLP 2023. [Paper](https://aclanthology.org/2023.findings-emnlp.317.pdf)
- SDOH-NLI Dataset: [GitHub](https://github.com/google-research-datasets/SDOH-NLI/tree/main)

