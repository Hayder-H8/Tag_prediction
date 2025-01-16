import os
from utils import build_dataset , show_tag_distr , co_occ_matrix ,corr_difficulty,clean_text,extract_top_5_word_embeddings ,tags_to_binary_vector , hamming_loss
import numpy as np
import pandas as pd
import nltk
import torch
import sys
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MultiLabelNN
from training import train_and_evaluate  ,cross_validate_threshold  , train_multiple_classifier_chains,train_svm_classifiers ,cross_validate_svm_threshold ,train_and_evaluate_random_forest
from utils import get_top_tfidf_words , get_glove_embeddings


def main():
    """
    here I try several methods (logistic regression , svms , chained classifiers) to learn from features extracted from textual description of
    exercices using learning from whole corpus ,I extract keywords based on tfidf scores, I also cross validate to get best thresholds

   
    """
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
                         
    
    print("preparing cleaned text")                     
    dataset['cleaned_text'] = dataset['text_input'].apply(clean_text)
    dataset['tag_vector'] = dataset['tags'].apply(lambda x: tags_to_binary_vector(x, valid_tags))


    corpus = dataset['cleaned_text'].tolist()
    sentences=[text.split() for text in corpus]
    print('lemmatizing......')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    sentences = [
    [lemmatizer.lemmatize(word) for word in text.split()] 
    for text in corpus

    model = Word2Vec(sentences, vector_size=20, window=5, min_count=1, sg=1, epochs=10)

    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_feature_names)


    top_words_list = []
    for _, row in tfidf_scores.iterrows():
        top_words = get_top_tfidf_words(row, tfidf_feature_names)
        top_words_list.append(top_words)
    dataset['top_words'] = top_words_list


    dataset['embeddings'] = dataset['top_words'].apply(lambda words: get_glove_embeddings(words, model))
    feature_array = np.stack([np.stack([tensor.item() for tensor in row]) for row in dataset['embeddings']])
    tag_array= dataset['tag_vector'].to_numpy()
    tag_array = np.array([np.array(x) for x in tag_array], dtype=np.float32)
    tags = torch.tensor(tag_array, dtype=torch.float32)
    features = torch.tensor(feature_array, dtype=torch.float32)
    n_inputs = features.size()[1]  
    n_outputs = tags.size()[1]  
    model = MultiLabelNN(n_inputs, n_outputs)
    f1,precision, accuracy, h_loss=train_and_evaluate(model,features,tags)
    X = features.numpy()
    Y = tags.numpy()
    thresholds=[0.3,0.4,0.5,0.6,0.7,0.8]
    print("**************cross validation to find best threshold for linear regression***************")
    cross_validate_threshold(X, Y, thresholds, num_splits=5)
    
    print("**************cross validation to find best threshold for svm***************")    
    cross_validate_svm_threshold(X, Y, thresholds, num_splits=5)
    train_multiple_classifier_chains(X, Y, threshold=0.5, num_chains=5)
    
    print("**************predict with random forests***************")    

    rf_models, rf_metrics = train_and_evaluate_random_forest(X, Y)
    
    return 0 


if __name__ == "__main__":
    main()



