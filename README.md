The project involves enhancing a WordLSTM model for text generation by introducing advanced features such as bidirectional LSTM layers, 
customizable embedding dimensions, dynamic dropout rates, and improved weight initialization techniques(Xavier uniform). These enhancements 
aim to improve the model's ability to understand context, manage variable input lengths, and efficiently train on large datasets. The goal is to create 
a more robust and flexible neural network capable of generating text with higher accuracy, better handling long-range dependencies, and 
adapting to different text generation tasks.

Contents:
Import Libraries
Load Dataset
3 Preprocessing and Exploring Text Data
3.1 Text Cleaning
3.2 Finding Word Count
3.3 Find and Replace Rare Words with "Unknown" Token
4 Data Preparation
4.1 Prepare Sequences
4.2 Create Token-Integer Mappings
4.3 Split Data into Train and Validation Sets
4.4 Pad Sequences
4.5 Convert Text Sequences to Integer Sequences
Model Building
5.1 Define Model Architecture
5.2 Start Model Training
Text Generation
