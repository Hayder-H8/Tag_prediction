# Tag Prediction

This repository contains the code and models developed for the Illuin Tech Challenge, focused on predicting tags for exercises based on textual and code features.

## Overview

The goal of this project is to build models that predict tags for a given exercise using features extracted from either:
1. **Textual Descriptions** of the exercise.
2. **Source Code** of the exercise.

### Feature Extraction
Two methods are implemented for feature extraction:
- **BERT Embeddings**: Pre-trained language models are used to encode textual descriptions.
- **Learned Embeddings**: Custom embeddings are trained on the provided corpus.

### Model Training and Prediction
The extracted features are used to train various models, including:
- **Linear Regressors**
- **Support Vector Machines (SVMs)**
- **Random Forest Classifiers**
- **Simple Deep Learning Models**

### Workflow
1. **Feature Extraction**: Features are extracted from either the exercise descriptions or their code.
2. **Model Training**: Different models are trained using the extracted features.
3. **Prediction**: Models are evaluated on their ability to predict tags.

## How to Use

### Extract Features and Train Models
To extract features from the source code of exercises, train models, and make predictions using linear regressors, random forests, and SVMs, run the following command:

```bash
python code_embeds.py

