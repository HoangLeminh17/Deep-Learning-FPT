# Deep-Learning-FPT
# Projects on class

## Convolution step by step
- This notebook is based on an assignment from the [Convolutional Neural Networks course](https://www.coursera.org/learn/convolutional-neural-networks) by DeepLearning.AI. 
- I completed the implementation as part of my own learning process.  
- Modifications, comments, and explanation are my own. **For educational purposes only.**
- This notebook implements forward propagration, backpropragation of volume convolution: 1 layer (Input -> Conv -> Pooling -> Output)
## Sentiment Analysis - Binary classification
- Using the dataset ibmdb_reviews which containts 50,000 reviews films, splitted into 25k train and 25k test, labels are 0 or 1 as negative or positive
- Techniques used: Word embedding, bidirectional LSTM, Dropout, L2 Regularization
- Steps:
1. **Load, process data:**  
  1.1. Use tensorflow_datasets to load the **"ibmdb_reviews"**  
  1.2. Split into train, test  
  1.3. Tokenize, truncating, padding sentences
2. **Build sequential model and train:**    
  2.1 Embedding, Bidirectional(LSTM), Dropout, Dense(ReLU), Output(Sigmoid)   
  2.2.Loss: binary_crossentropy, optimizer: Adam
3. **Evaluation** uses accuracy
   
## Project 1: Hyperparameters tuning
- I try to achieve better metrics by using algorithms that search for hyperparameters, such as Bayes Optimization, Genetic Algorithm
- Genetic Algorithm requires a **REALLY** long time for searching and a strong device to train, recommend for the Bayes 
