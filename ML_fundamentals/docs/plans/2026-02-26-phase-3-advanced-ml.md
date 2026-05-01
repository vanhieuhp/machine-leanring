# Phase 3: Advanced ML Techniques - Implementation Plan

> **For Claude:** Use the tools directly to implement this plan task-by-task.

**Goal:** Create comprehensive learning materials for Phase 3 Advanced ML including 4 topics (Ensemble Methods, SVM, Neural Networks, NLP Basics) with guides, code examples, exercises, and final project.

**Architecture:** Follow Phase 1 & 2 structure - each topic has: guide.md, numbered code files (01_, 02_, etc.), and exercises.py. Match the depth and style from Phase 1 & 2.

**Tech Stack:** Python, scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, NLTK, spaCy

---

## Phase 3 Structure Overview

```
Phase_3_Advanced/
├── PHASE_3_GUIDE.md (update with links)
├── 01_Ensemble_Methods/
│   ├── guide.md
│   ├── 01_random_forest.py
│   ├── 02_gradient_boosting.py
│   ├── 03_xgboost_lightgbm.py
│   ├── 04_stacking_voting.py
│   └── exercises.py
├── 02_SVM/
│   ├── guide.md
│   ├── 01_svm_classification.py
│   ├── 02_svm_regression.py
│   ├── 03_kernel_methods.py
│   └── exercises.py
├── 03_Neural_Networks/
│   ├── guide.md
│   ├── 01_perceptron.py
│   ├── 02_keras_basics.py
│   ├── 03_cnn_basics.py
│   ├── 04_rnn_basics.py
│   ├── 05_advanced_architectures.py
│   └── exercises.py
├── 04_NLP_Basics/
│   ├── guide.md
│   ├── 01_text_preprocessing.py
│   ├── 02_word_embeddings.py
│   ├── 03_sentiment_analysis.py
│   ├── 04_text_classification.py
│   └── exercises.py
└── Project_Advanced_ML/
    ├── README.md
    └── src/
        ├── data_loader.py
        ├── model.py
        ├── trainer.py
        └── evaluator.py
```

---

## Task 1: Update PHASE_3_GUIDE.md with Study Path

**Files:**
- Modify: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\PHASE_3_GUIDE.md`

**Step 1: Add Study Path section to guide**

Update the PHASE_3_GUIDE.md to include detailed study path showing the files for each topic. Add this after line 46 (after Topics section):

```markdown
## Study Path

### Topic 1: Ensemble Methods (Week 1-2)
- Start with: `01_Ensemble_Methods/guide.md` - Read theory
- Then: `01_Ensemble_Methods/01_random_forest.py` - Learn Random Forest
- Next: `01_Ensemble_Methods/02_gradient_boosting.py` - Gradient Boosting
- Next: `01_Ensemble_Methods/03_xgboost_lightgbm.py` - XGBoost & LightGBM
- Next: `01_Ensemble_Methods/04_stacking_voting.py` - Ensemble techniques
- Practice: `01_Ensemble_Methods/exercises.py`

### Topic 2: Support Vector Machines (Week 3)
- Start with: `02_SVM/guide.md` - Read theory
- Then: `02_SVM/01_svm_classification.py` - SVM classification
- Next: `02_SVM/02_svm_regression.py` - SVM regression
- Next: `02_SVM/03_kernel_methods.py` - Kernel tricks
- Practice: `02_SVM/exercises.py`

### Topic 3: Neural Networks (Week 4-5)
- Start with: `03_Neural_Networks/guide.md` - Read theory
- Then: `03_Neural_Networks/01_perceptron.py` - Perceptron & backprop
- Next: `03_Neural_Networks/02_keras_basics.py` - Keras fundamentals
- Next: `03_Neural_Networks/03_cnn_basics.py` - CNN for images
- Next: `03_Neural_Networks/04_rnn_basics.py` - RNN for sequences
- Next: `03_Neural_Networks/05_advanced_architectures.py` - Modern NNs
- Practice: `03_Neural_Networks/exercises.py`

### Topic 4: NLP Basics (Week 6)
- Start with: `04_NLP_Basics/guide.md` - Read theory
- Then: `04_NLP_Basics/01_text_preprocessing.py` - Text cleaning
- Next: `04_NLP_Basics/02_word_embeddings.py` - Word vectors
- Next: `04_NLP_Basics/03_sentiment_analysis.py` - Sentiment analysis
- Next: `04_NLP_Basics/04_text_classification.py` - Text classification
- Practice: `04_NLP_Basics/exercises.py`

### Final Project
- Start with: `Project_Advanced_ML/README.md` - Project overview
- Implement: Build an advanced ML model
```

---

## Task 2: Create Ensemble Methods Topic

**Files:**
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\01_Ensemble_Methods\guide.md`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\01_Ensemble_Methods\01_random_forest.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\01_Ensemble_Methods\02_gradient_boosting.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\01_Ensemble_Methods\03_xgboost_lightgbm.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\01_Ensemble_Methods\04_stacking_voting.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\01_Ensemble_Methods\exercises.py`

### Step 1: Create guide.md for Ensemble Methods

Write comprehensive guide covering:
- What is ensemble learning
- Bagging vs Boosting vs Stacking
- Random Forest deep dive
- Gradient Boosting concepts
- XGBoost & LightGBM
- When to use each method
- Key hyperparameters
- Advantages & disadvantages

### Step 2: Create 01_random_forest.py

Include:
- Decision tree review
- Random Forest concept (bagging)
- Building RF from scratch
- Using scikit-learn
- Hyperparameters (n_estimators, max_depth, min_samples_split)
- Feature importance
- OOB score
- Practical example with dataset

### Step 3: Create 02_gradient_boosting.py

Include:
- Boosting concept
- Gradient descent analogy
- Gradient Boosting algorithm
- Using scikit-learn GradientBoostingClassifier/Regressor
- Hyperparameters (learning_rate, n_estimators, max_depth)
- Overfitting prevention
- Comparison with Random Forest

### Step 4: Create 03_xgboost_lightgbm.py

Include:
- XGBoost introduction
- XGBoost vs sklearn GradientBoosting
- Key hyperparameters
- LightGBM introduction
- Speed comparison
- Practical examples with both
- Early stopping
- Cross-validation with XGBoost

### Step 5: Create 04_stacking_voting.py

Include:
- VotingClassifier (hard vs soft)
- Stacking concepts
- Building stacked ensemble
- Using scikit-learn StackingClassifier
- Meta-learner selection
- Real-world example combining multiple models

### Step 6: Create exercises.py

Include 5-7 exercises:
1. Compare Random Forest vs Gradient Boosting on same dataset
2. Tune XGBoost hyperparameters
3. Build voting ensemble
4. Analyze feature importance
5. Handle imbalanced data with ensemble
6. Stack multiple models
7. Compare performance metrics

---

## Task 3: Create SVM Topic

**Files:**
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\02_SVM\guide.md`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\02_SVM\01_svm_classification.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\02_SVM\02_svm_regression.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\02_SVM\03_kernel_methods.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\02_SVM\exercises.py`

### Step 1: Create guide.md for SVM

Write comprehensive guide covering:
- What is SVM
- Hyperplane and margin
- Support vectors
- Linear SVM
- Non-linear SVM with kernels
- Kernel functions (linear, polynomial, RBF)
- C parameter and gamma
- SVM for regression
- Advantages & disadvantages

### Step 2: Create 01_svm_classification.py

Include:
- Linear SVM concept
- Maximum margin classifier
- Using SVC from sklearn
- Hyperparameter C (regularization)
- Practical classification example
- Visualization of decision boundary

### Step 3: Create 02_svm_regression.py

Include:
- SVM for regression (SVR)
- Epsilon-insensitive loss
- Using SVR from sklearn
- Hyperparameter epsilon
- Practical regression example
- Compare with linear regression

### Step 4: Create 03_kernel_methods.py

Include:
- Kernel trick explanation
- Polynomial kernel
- RBF (Gaussian) kernel
- Choosing kernel parameters
- Grid search for SVM
- Comparing kernels on non-linear data
- Visualization of kernel transformations

### Step 5: Create exercises.py

Include 5 exercises:
1. Classify iris data with SVM
2. Compare linear vs RBF kernel
3. Tune C and gamma parameters
4. Use SVM for regression
5. Visualize decision boundaries

---

## Task 4: Create Neural Networks Topic

**Files:**
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\03_Neural_Networks\guide.md`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\03_Neural_Networks\01_perceptron.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\03_Neural_Networks\02_keras_basics.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\03_Neural_Networks\03_cnn_basics.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\03_Neural_Networks\04_rnn_basics.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\03_Neural_Networks\05_advanced_architectures.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\03_Neural_Networks\exercises.py`

### Step 1: Create guide.md for Neural Networks

Write comprehensive guide covering:
- What are neural networks
- Perceptron and activation functions
- Multi-layer perceptron
- Forward and backward propagation
- Loss functions
- Optimizers (SGD, Adam, etc.)
- CNN for images
- RNN for sequences
- Modern architectures overview

### Step 2: Create 01_perceptron.py

Include:
- Single neuron concept
- Perceptron algorithm
- Activation functions (step, sigmoid, ReLU)
- Implementing perceptron from scratch
- XOR problem
- Multi-layer perceptron concept

### Step 3: Create 02_keras_basics.py

Include:
- Keras/Sequential API
- Dense layers
- Compiling model (loss, optimizer)
- Training (fit method)
- Callbacks (early stopping, checkpoint)
- Model evaluation
- Saving/loading models
- MNIST classification example

### Step 4: Create 03_cnn_basics.py

Include:
- Convolutional layers
- Pooling layers
- Flatten layer
- Building CNN architecture
- LeNet-style CNN
- Image classification with CNN
- Data augmentation basics

### Step 5: Create 04_rnn_basics.py

Include:
- Recurrent layers
- RNN for sequences
- LSTM explanation
- GRU explanation
- Sequence prediction example
- Text generation basics

### Step 6: Create 05_advanced_architectures.py

Include:
- ResNet skip connections
- Inception concepts
- Transformer basics (attention)
- BERT/GPT concepts
- Transfer learning
- Using pre-trained models

### Step 7: Create exercises.py

Include 7 exercises:
1. Build MLP for MNIST
2. Experiment with activation functions
3. Build CNN for image classification
4. Add dropout to reduce overfitting
5. Build LSTM for sequence prediction
6. Use transfer learning
7. Implement early stopping

---

## Task 5: Create NLP Basics Topic

**Files:**
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\04_NLP_Basics\guide.md`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\04_NLP_Basics\01_text_preprocessing.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\04_NLP_Basics\02_word_embeddings.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\04_NLP_Basics\03_sentiment_analysis.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\04_NLP_Basics\04_text_classification.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\04_NLP_Basics\exercises.py`

### Step 1: Create guide.md for NLP Basics

Write comprehensive guide covering:
- What is NLP
- Text preprocessing
- Bag of Words
- TF-IDF
- Word embeddings
- Word2Vec, GloVe
- Sentiment analysis
- Text classification
- Modern NLP (transformers)

### Step 2: Create 01_text_preprocessing.py

Include:
- Lowercasing
- Tokenization
- Stopword removal
- Stemming
- Lemmatization
- Regular expressions
- Using NLTK
- Using spaCy basics

### Step 3: Create 02_word_embeddings.py

Include:
- Bag of Words
- TF-IDF
- Word embeddings concept
- Word2Vec (CBOW, Skip-gram)
- Using pre-trained embeddings
- Loading GloVe vectors
- Embedding layers in Keras

### Step 4: Create 03_sentiment_analysis.py

Include:
- Sentiment analysis concepts
- Building pipeline
- Using TF-IDF + classifier
- Using word embeddings + NN
- Real dataset (IMDb, Twitter)
- Evaluation metrics
- Practical example

### Step 5: Create 04_text_classification.py

Include:
- Text classification overview
- Multi-class classification
- Neural network approach
- Padding and truncating
- Building classifier with Keras
- Using pre-trained models
- Real-world example

### Step 6: Create exercises.py

Include 6 exercises:
1. Preprocess text data
2. Create TF-IDF features
3. Train classifier for sentiment
4. Build text classifier with NN
5. Use pre-trained embeddings
6. Compare different approaches

---

## Task 6: Create Advanced ML Project

**Files:**
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\Project_Advanced_ML\README.md`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\Project_Advanced_ML\src\data_loader.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\Project_Advanced_ML\src\model.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\Project_Advanced_ML\src\trainer.py`
- Create: `c:\Users\hieunv\Workspaces\Myself\Projects\machine-leanring\ML_fundamentals\Phase_3_Advanced\Project_Advanced_ML\src\evaluator.py`

### Step 1: Create README.md

Write project overview:
- Project options
- Prerequisites
- Learning objectives
- Dataset recommendations
- Expected deliverables

### Step 2: Create src files

Create modular structure:
- data_loader.py: Load and preprocess data
- model.py: Define models (ensemble, NN, or NLP)
- trainer.py: Training logic
- evaluator.py: Evaluation and metrics

---

## Summary

This plan creates:
- 4 comprehensive topics with guides
- 21 code files with detailed examples
- 4 exercise files
- 1 final project structure

Total: ~30 files following Phase 1 & 2 patterns
