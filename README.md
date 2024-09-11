### SmartComplete

**Data Preparation**
The Taskmaster dialogues were cleaned to remove unwanted characters, expand contractions, and remove less frequent words to avoid overfitting the model. I used n-gram preprocessing to tokenize the cleaned text and create token-to-integer mappings, essential for the LSTM model. This approach helps the model understand word sequences and predict the next token based on the dialogue context.

**Model Development**
The LSTM model was built with an embedding layer, LSTM layers to process sequential data, and a fully connected layer for token prediction. The model was trained using the preprocessed data, optimized with cross-entropy loss, and regularized with dropout to prevent overfitting. Its performance was evaluated using perplexity to measure prediction accuracy.

**Model Deployment**
The model was deployed as a RESTful API using Flask. Users can submit a seed text to the API, which responds with a completed dialogue sequence. The app was designed with a user-friendly web interface and deployed on Google Cloud Platform (GCP) for scalability and easy access.
