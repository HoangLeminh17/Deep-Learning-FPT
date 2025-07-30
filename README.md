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
- This project is comparison on optimizers (Bayes Optimization and Genetic algorithm) on the Fashion MNIST dataset
- Genetic Algorithm requires a **really** long time for searching and a strong device to train, recommend for the Bayes

## Project 2: Deepfake detection
- My team and I used pre-trained ResNet50 to distinguish between real and deepfake images with a pretty number ~ 81% on test data
- Architecture: Based Model (ResNet50) -> Pooling -> Densely -> Output
- For more details, check on Project 2 - DPL.pdf (Vietnamese), slide: Deepfake detection.pdf
- Link dataset: https://drive.google.com/file/d/1HnrO5sGg8gc2VuC1dUHE_62WIbYMSXid/view?usp=sharing
- Link model.h5: https://drive.google.com/file/d/1LF-fYG8r5gXuvHmEEA2y8xLyKeE7Z5pB/view?usp=sharing

## Project 3: Speech emtion recognition
- With the same architecture (2 x CNN -> LSTM -> Output), same distribution of train - test set, this aims to compare performance model using normal data vs augmentation data
- Datasets used: Crema_D, RAVDESS, SAVEE, TESS. These 4 datasets express emotion through voice, not through meaning of words.
- For more details, check on DPL-Project-3- Report.pdf (Vietnamese), slide: Project 3 - DPL.pdf
- Link dataset: https://drive.google.com/file/d/1jS8GiWf2PMS_91gGcjCwESsA0MSuyOT9/view?usp=sharing

