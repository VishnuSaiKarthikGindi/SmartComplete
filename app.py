from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import torch
from torch import nn
import torch.nn.functional as F

with open('Dailog-dataset/vocab/int2token.json', 'r') as f:
    int2token = json.load(f)
with open('Dailog-dataset/vocab/token2int.json', 'r') as f:
    token2int = json.load(f)

vocab_size = len(int2token)

class WordLSTM(nn.Module):
    def __init__(self, embedding_dim=200, n_hidden=256, n_layers=2, drop_prob=0.3, lr=0.001, bidirectional=False):
        super(WordLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, n_hidden, n_layers, batch_first=True, dropout=drop_prob, bidirectional=bidirectional)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(n_hidden * 2 if bidirectional else n_hidden, vocab_size)
        self.init_weights()

    def forward(self, x, hidden):
        embedded = self.embed(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        out = self.dropout(lstm_out)
        out = out.reshape(-1, self.n_hidden * 2 if self.bidirectional else self.n_hidden)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 2 if self.bidirectional else 1
        return (weight.new(self.n_layers * number, batch_size, self.n_hidden).zero_().to(weight.device),
                weight.new(self.n_layers * number, batch_size, self.n_hidden).zero_().to(weight.device))

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

net = WordLSTM()  
net.load_state_dict(torch.load('saved_weights.pt', map_location=torch.device('cpu')))
#net.eval()

def predict(net, tkn, h=None):
    x = np.array([[token2int[tkn]]])
    inputs = torch.from_numpy(x)

    if torch.cuda.is_available():
        inputs = inputs.cuda()
    out, h = net(inputs, h)
    p = F.softmax(out, dim=1).data

    if torch.cuda.is_available():
        p = p.cpu()

    p = p.numpy()
    sampled_token_index = np.argmax(p, axis=1)[0]
    return int2token[sampled_token_index], h

def sample(net, size=2, seed_text='it is'):
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    h = net.init_hidden(1)  # Initialize the hidden state for LSTM
    toks = seed_text.split()

    # Predict the next token based on the seed text
    for t in toks:
        token, h = predict(net, t, h)
    toks.append(token)

    # Continue generating the rest of the sequence
    for i in range(size - 1):
        token, h = predict(net, toks[-1], h)
        toks.append(token)

    return ' '.join(toks)

# Create the Flask app
app = Flask(__name__)
CORS(app)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.json
    seed_text = data.get('seed_text', '')
    num_tokens = data.get('num_tokens', 6)

    if not seed_text:
        return jsonify({"error": "Seed text is required"}), 400

    # Generate the text using the LSTM model
    try:
        generated_text = sample(net, size=num_tokens, seed_text=seed_text)
        return jsonify({"seed_text": seed_text, "generated_text": generated_text})
    except KeyError as e:
        return jsonify({"error": f"Unknown token in seed text: {e}"}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
