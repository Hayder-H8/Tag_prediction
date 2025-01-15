import os
from utils import build_dataset , show_tag_distr , co_occ_matrix ,corr_difficulty,clean_text,extract_top_5_word_embeddings ,tags_to_binary_vector , hamming_loss
import numpy as np
import torch
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MultiLabelNN
from training import train_and_evaluate  ,cross_validate_threshold ,train_multiple_classifier_chains,cross_validate_svm_threshold


def main():
    """here I try several methods(logistic regression , svms , chained classifiers) to learn from features extracted from textual description of
    exercices using a small bert model. I extract keywords based on cosine similarity , I also cross validate to get best thresholds

    _
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
    
    embeddings = []
    print('preparing embeddings , this might take over 20 min..sorry')
    # Loop over the dataset rows to process the cleaned text
    for idx, text in dataset['cleaned_text'].items():
        try:
            # Extract flattened top 5keyword embeddings for each text
            embedding = extract_top_5_word_embeddings(text)
            embeddings.append(embedding)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            embeddings.append(None)  
    print('finished')

    dataset['text_embeds'] = embeddings
    
    dataset['tag_vector'] = dataset['tags'].apply(lambda x: tags_to_binary_vector(x, valid_tags))
    dataset = dataset[dataset['text_embeds'].notnull()].reset_index(drop=True)
    numpy_array = np.stack([np.stack([tensor.item() for tensor in row]) for row in dataset['text_embeds']])
    features = torch.tensor(numpy_array, dtype=torch.float32)
    tag_array= dataset['tag_vector'].to_numpy()
    tag_array = np.array([np.array(x) for x in tag_array], dtype=np.float32)
    tags = torch.tensor(tag_array, dtype=torch.float32)
    
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

    
    return 0

if __name__ == "__main__":
    main()