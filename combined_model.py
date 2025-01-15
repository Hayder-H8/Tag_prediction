import os
from utils import build_dataset , show_tag_distr,get_top_tfidf_words , get_glove_embeddings , co_occ_matrix ,corr_difficulty,clean_text,extract_top_5_word_embeddings ,tags_to_binary_vector , hamming_loss ,combine_predictions ,evaluate_nn_model, evaluate_lr_models
import numpy as np
import torch
import sys
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import argparse
from sklearn.metrics import hamming_loss, precision_score, recall_score, f1_score
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MultiLabelNN
from training import train_and_evaluate  ,cross_validate_threshold ,train_multiple_classifier_chains,cross_validate_svm_threshold , train_logistic_regression_classifiers


def get_args():
    parser = argparse.ArgumentParser(description="Combine predictions from two models (NN and Logistic Regression).")
    parser.add_argument("--method", type=str, default="soft_voting", choices=["soft_voting", "hard_voting"],
                        help="Combination method to use: 'soft_voting' or 'hard_voting'.")
    parser.add_argument("--threshold_nn", type=float, default=0.5, 
                        help="Threshold to apply for nn.")
    parser.add_argument("--threshold_lr", type=float, default=0.5, 
                        help="Threshold to apply for binary classification.")
    parser.add_argument("--threshold", type=float, default=0.5, 
                        help="Threshold to apply for combination.")
    
    parser.add_argument("--epochs", type=int, default=100, 
                        help="Number of epochs to train the neural network model.")
    return parser.parse_args()

def main():
    """
    Here I try to extract code embeddings with code bert and feed it to a linear DL model 
    and combine the predicted output with 8 Linear regressors that learn from text
    features learned via the whole corpus of textual description of the problem
    
    """
    
    args = get_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    folder_path = './code_classification_dataset'
    dataset = build_dataset(folder_path)
    print ("the dataset is of shape", dataset.shape)
    # list of 8 valid tags 
    valid_tags = ['math', 'graphs', 'strings', 'number theory', 'trees', 'geometry', 'games', 'probabilities']
    # Filter tags for each row
    dataset['tags'] = dataset['tags'].apply(lambda taglist: [tag for tag in taglist if tag in valid_tags])
    #show_tag_distr(dataset)
    dataset = dataset[dataset["tags"].apply(lambda x: len(x) > 0)]
    #co_occ_matrix(dataset,valid_tags)
    #corr_difficulty(dataset)
    dataset['text_input'] = dataset['prob_desc_description'].fillna('') + " " + \
                         dataset['prob_desc_notes'].fillna('') + " " + \
                         dataset['prob_desc_output_spec'].fillna('')
                         
    
    print("preparing cleaned text ......")                     
    dataset['cleaned_text'] = dataset['text_input'].apply(clean_text)
    
    corpus = dataset['cleaned_text'].tolist()
    sentences=[text.split() for text in corpus]

    model = Word2Vec(sentences, vector_size=20, window=5, min_count=1, sg=1, epochs=10)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)

    print("selecting top keywords ......")
    top_words_list = []
    for _, row in tfidf_scores.iterrows():
        top_words = get_top_tfidf_words(row, tfidf_feature_names)
        top_words_list.append(top_words)
    dataset['top_words'] = top_words_list


    dataset['embeddings'] = dataset['top_words'].apply(lambda words: get_glove_embeddings(words, model))
    
    dataset['tag_vector'] = dataset['tags'].apply(lambda x: tags_to_binary_vector(x, valid_tags))
    numpy_array = np.stack([np.stack([tensor.item() for tensor in row]) for row in dataset['embeddings']])
    features = torch.tensor(numpy_array, dtype=torch.float32)
    tag_array= dataset['tag_vector'].to_numpy()
    tag_array = np.array([np.array(x) for x in tag_array], dtype=np.float32)
    tags = torch.tensor(tag_array, dtype=torch.float32)
    
    
    
    print("embedding code using code bert ......")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaModel.from_pretrained("microsoft/codebert-base")
    model.to(device)
    def compute_embedding(code):
        inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        
        with torch.no_grad():  
            outputs = model(**inputs)
        
        embeddings = outputs.last_hidden_state
        cls_embedding = embeddings[:, 0, :]  
        return cls_embedding.cpu().numpy()  

    dataset['code_embed'] = dataset['source_code'].apply(compute_embedding)
    numpy_array2 = np.stack([np.array(row) for row in dataset['code_embed']])
    numpy_array2 = np.squeeze(numpy_array2, axis=1)
    features2 = torch.tensor(numpy_array2, dtype=torch.float32)
    print("predicting using code bert embedding and DL learned model")
    n_inputs = features2.size()[1]  
    n_outputs = tags.size()[1]  
    model = MultiLabelNN(n_inputs, n_outputs)
    f1,precision, accuracy, h_loss=train_and_evaluate(model,features2,tags,num_epochs=100)
    
    
    X = features.numpy()
    Y = tags.numpy()
    models,_ =train_logistic_regression_classifiers(X, Y, threshold=args.threshold_lr)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    print("predicting using logistic regressions and keyword learned embeddings")
    lr_predictions = evaluate_lr_models(models, X_test, threshold=args.threshold_lr)
    print("lr hamming loss is :  "  , hamming_loss(Y_test,lr_predictions))

    # Neural network evaluation
    train_size = int(0.8 * len(features))  # 80% for training
    X_test_tensor = features2[train_size:]
    y_test_tensor = tags[train_size:]
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


    
    y_true, nn_predictions = evaluate_nn_model(model, test_loader, threshold=args.threshold_nn)
    print("nn hamming loss is :  "  , hamming_loss(Y_test,nn_predictions))


    # Combine predictions
    combined_predictions = combine_predictions(nn_predictions, lr_predictions, method='hard_voting', threshold=args.threshold)

    

    hamming = hamming_loss(Y_test, combined_predictions)
    per_class_precision = precision_score(Y_test, combined_predictions, average=None,zero_division=0)
    per_class_recall = recall_score(Y_test, combined_predictions, average=None,zero_division=0)
    per_class_f1 = f1_score(Y_test, combined_predictions, average=None,zero_division=0)
    
    print(f"Hamming Loss: {hamming:.4f}")

    print("\nPer-Class Metrics:")
    for i, (p, r, f1) in enumerate(zip(per_class_precision, per_class_recall, per_class_f1)):
        print(f"Class {i}: Precision={p:.4f}, Recall={r:.4f}, F1-Score={f1:.4f}")
    return 0

if __name__ == "__main__":
    main()



    