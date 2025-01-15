import json
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)


def build_dataset(folder_path):
    
    dataset = []

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jsonl') or file_name.endswith('.json'):
            file_id = int(file_name.split('_')[1].split('.')[0])
            
            with open(os.path.join(folder_path, file_name), 'r' , encoding='utf-8') as file:
                data = json.load(file)
                
                data['id'] = file_id
                
                dataset.append(data)

    return pd.DataFrame(dataset)

def print_unique_tags(dataset):
    unique_tags = set(tag for tags in dataset['tags'] for tag in tags)
    print("Unique tags:", unique_tags)

def show_tag_distr(dataset):
    all_tags = [tag for tags in dataset['tags'] for tag in tags]
    tag_counts = Counter(all_tags)
    # Convert to a DataFrame for visualization
    tag_counts_df = pd.DataFrame(tag_counts.items(), columns=['Tag', 'Count']).sort_values(by='Count', ascending=False)

# Plot the distribution
    plt.figure(figsize=(10, 6))
    plt.bar(tag_counts_df['Tag'], tag_counts_df['Count'], color='skyblue')
    plt.xlabel('Tags')
    plt.ylabel('Count')
    plt.title('Distribution of Tags')
    plt.xticks(rotation=45)
    plt.show()
    
def co_occ_matrix(dataset , valid_tags):
    co_occurrence_matrix = pd.DataFrame(
    data=0,
    index=valid_tags,
    columns=valid_tags
)

    for tags in dataset['tags']:
        for i in range(len(tags)):
            for j in range(len(tags)):
                co_occurrence_matrix.loc[tags[i], tags[j]] += 1


    plt.figure(figsize=(8, 6))
    sns.heatmap(co_occurrence_matrix, annot=True, fmt="d", cmap="Blues")
    plt.title("Tag Co-occurrence Matrix")
    plt.show()




    
def corr_difficulty(dataset):
    tags_df = dataset['tags'].explode().reset_index()  
    tags_df = pd.get_dummies(tags_df['tags']).groupby(tags_df['index']).sum()  

    combined_df = pd.concat([dataset[['difficulty']], tags_df], axis=1)

    correlation = combined_df.corr()

    print(correlation['difficulty'].drop('difficulty').sort_values(ascending=False))




nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    cleaned_text = re.sub(r'\$\$\$.*?\$\$\$', '', text)  # Remove anything between $$$
    cleaned_text = re.sub(r'[^\w\s]', ' ', cleaned_text)  # Keep only alphanumeric characters and spaces
    cleaned_text = re.sub(r'\d+', '', cleaned_text)  # Remove digits

    tokens = word_tokenize(cleaned_text)

    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [word for word in tokens if word.lower() not in stop_words]

    cleaned_text = ' '.join(cleaned_tokens)
    
    return cleaned_text



def extract_top_5_word_embeddings(text):
    # Load the pre-trained BERT model and tokenizer with reduced hidden size
    tokenizer = BertTokenizer.from_pretrained('prajjwal1/bert-mini')
    model = BertModel.from_pretrained('prajjwal1/bert-mini')
    encoded_input = tokenizer(text, return_tensors='pt')
    
    with torch.no_grad():
        output = model(**encoded_input)
    

    last_hidden_state = output.last_hidden_state
    cls_embedding = last_hidden_state[0, 0].unsqueeze(0)  
    token_embeddings = last_hidden_state[0, 1:-1] 
    
    # Calculate cosine similarity between [CLS] token and all token embeddings
    similarities = cosine_similarity(cls_embedding, token_embeddings)[0]
    
    top_5_indices = similarities.argsort()[-5:] 
    
    top_5_embeddings = token_embeddings[top_5_indices]
    
    concatenated_embeddings = torch.cat([top_5_embeddings], dim=0)  
    
    return concatenated_embeddings.flatten()


def tags_to_binary_vector(tags, all_tags):
    binary_vector = np.zeros(len(all_tags))
    for tag in tags:
        if tag in all_tags:
            binary_vector[all_tags.index(tag)] = 1
    return binary_vector


def hamming_loss(y_true, y_pred):
        temp = 0
        for i in range(y_true.shape[0]):
            temp += np.size(y_true[i] == y_pred[i]) - np.count_nonzero(y_true[i] == y_pred[i])
        return temp / (y_true.shape[0] * y_true.shape[1])



def get_top_tfidf_words(row, feature_names, top_n=5):
    scores = row.to_numpy()
    indices = np.argsort(scores)[-top_n:][::-1]
    return [feature_names[i] for i in indices]

def get_glove_embeddings(words, model):
    embeddings = []
    for word in words:
        if word in model.wv:  
            embeddings.append(model.wv[word])
        else:
            embeddings.append(np.zeros(model.vector_size))  
    return np.concatenate(embeddings) if embeddings else np.zeros(model.vector_size)





# Evaluate the deep learning model
def evaluate_nn_model(model, test_loader, threshold=0.5):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            predicted = (outputs > threshold).float()  
            y_true.append(batch_y)
            y_pred.append(predicted)

    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return y_true, y_pred

# Evaluate the logistic regression models
def evaluate_lr_models(models, X_test, threshold=0.5):
    all_predictions = []
    for i, model in enumerate(models):
        probabilities = model.predict_proba(X_test)[:, 1]  
        predictions = (probabilities >= threshold).astype(int)  
        all_predictions.append(predictions)

    return np.array(all_predictions).T

# Combine 
def combine_predictions(nn_predictions, lr_predictions, method='soft_voting', threshold=0.5):
    if method == 'soft_voting':
        # Average the predictions (probabilities)
        combined_probabilities = (nn_predictions + lr_predictions) / 2
        combined_predictions = (combined_probabilities >= threshold).astype(int)
    elif method == 'hard_voting':
        # Majority voting (binary predictions)
        combined_predictions = (nn_predictions + lr_predictions) >= 2  # At least 2 votes for '1'
        combined_predictions = combined_predictions.astype(int)
    else:
        raise ValueError("Invalid combination method. Choose 'soft_voting' or 'hard_voting'.")

    return combined_predictions





