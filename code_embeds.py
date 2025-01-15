import os
from utils import build_dataset , show_tag_distr , co_occ_matrix ,corr_difficulty,clean_text,extract_top_5_word_embeddings ,tags_to_binary_vector , hamming_loss
import numpy as np
import torch
import sys
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import MultiLabelNN
from training import train_and_evaluate  ,cross_validate_threshold ,train_multiple_classifier_chains,cross_validate_svm_threshold , train_and_evaluate_random_forest


def main():
    """Here I try to extract code embeddings with code bert and feed it to a linear DL model 
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

    # Populate the 'code_embed' column
    dataset['code_embed'] = dataset['source_code'].apply(compute_embedding)
    numpy_array = np.stack([np.array(row) for row in dataset['code_embed']])
    numpy_array = np.squeeze(numpy_array, axis=1)
    features = torch.tensor(numpy_array, dtype=torch.float32)
    dataset['tag_vector'] = dataset['tags'].apply(lambda x: tags_to_binary_vector(x, valid_tags))
    tag_array= dataset['tag_vector'].to_numpy()
    tag_array = np.array([np.array(x) for x in tag_array], dtype=np.float32)
    tags = torch.tensor(tag_array, dtype=torch.float32)
    
    n_inputs = features.size()[1]  
    n_outputs = tags.size()[1]  
    model = MultiLabelNN(n_inputs, n_outputs)
    f1,precision, accuracy, h_loss=train_and_evaluate(model,features,tags,num_epochs=100)
    
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