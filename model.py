import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed_size= embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        ## turn captions into embeddings
        ## embeds shape: [batch, caption_length, embed_size]
        embeds = self.embed(captions)
        
        ## concat features and embeds along y-axis embeds as input of lstm
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        
        ## feed input into lstm
        lstm_out, _ = self.lstm(embeds)
        
        ## turn lstm output as prediction
        outputs = self.linear(lstm_out[:,:-1,:])
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        ## accepts pre-processed image tensor (inputs)
        ## returns predicted sentence (list ids as integer of length max_len)
        ids = []
        for i in range(max_len):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            _, predicted = outputs.max(1)
            ids.append(predicted.item()) ## predicted is a tensor and needs to be converted to int
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1) 
        return ids