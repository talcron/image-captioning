################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# Build and return the model here based on the configuration.
def get_model(config_data, vocab, network_block=''):
    """ vocab is Vocabulary() object from vocab.py """
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    p_dropout = config_data['generation']['deterministic']
    if not p_dropout:
        p_dropout = 0.0

    if network_block == 'encoder':
        return Encoder(hidden_size)
    elif network_block == 'decoder':
        return Decoder(hidden_size=hidden_size, embedding_size=embedding_size, vocab_size=len(vocab))
    elif network_block == 'stacked_decoder':
        return StackedDecoder(hidden_size=hidden_size, embedding_size=embedding_size, vocab_size=len(vocab),
                              dropout=p_dropout)
    elif network_block == 'RNNdecoder':
        return RNNDecoder(hidden_size=hidden_size, embedding_size=embedding_size, vocab_size=len(vocab))
    elif network_block == 'stacked_RNNdecoder':
        return StackedRNNDecoder(hidden_size=hidden_size, embedding_size=embedding_size, vocab_size=len(vocab),
                                 dropout=p_dropout)
    else:
        assert(1 == 0), 'must select network block in get_model()'


    # raise NotImplementedError("Model Factory Not Implemented")



class Encoder(nn.Module):
    def __init__(self, hidden_size):
        super(Encoder, self).__init__()

        self.resnet = torchvision.models.resnet50(pretrained=True, progress=True)
        for module in self.resnet.modules():
            if isinstance(module, nn.Conv2d):
                module.requires_grad_(requires_grad=False)
            if isinstance(module, nn.BatchNorm2d):
                module.requires_grad_(requires_grad=False)
        self.resnet.fc = nn.Linear(in_features=2048, out_features=hidden_size, bias=True)
        self.resnet.fc.requires_grad_(True)

    def forward(self, x):
        return self.resnet(x)


class Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.lstm = nn.LSTMCell(input_size=self.embedding_size, hidden_size=self.hidden_size)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

    def forward(self, x, captions, temperature=0.1, max_length=20, gen=False):
        """ x is Encoder() output, captions is (batch_size, max_sentence_len) """

        hidden_state = (x, x)
        if gen:
            start_captions = torch.ones((x.size(0), 1)).long()
            output = torch.empty((x.size(0), max_length)).long()
            word_embed = self.embed(start_captions)
            if temperature != 0:
                for t in range(0, max_length):
                    hidden_state = self.lstm(word_embed[:, 0, :], hidden_state)
                    output[:, t] = torch.squeeze(torch.multinomial(F.softmax(self.fc_out(hidden_state[0]) / temperature, dim=1), num_samples=1), dim=1)
                    word_embed = self.embed(torch.unsqueeze(output[:, t], dim=1))
            else:
                for t in range(0, max_length):
                    hidden_state = self.lstm(word_embed[:, 0, :], hidden_state)
                    output[:, t] = torch.argmax(self.fc_out(hidden_state[0]), dim=1)
                    word_embed = self.embed(torch.unsqueeze(output[:, t], dim=1))

            return output

        captions_embed = self.embed(captions)
        # shape (batch_size, max_sentence_length, embedding_dim)
        output = torch.empty((captions.size(0), captions.size(1), self.vocab_size))
        for t in range(captions.size(1)):

            hidden_state = self.lstm(captions_embed[:, t, :], hidden_state)

            output[:, t, :] = self.fc_out(hidden_state[0])

        return output


class StackedDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, num_layers=2, dropout=0.0):
        super(StackedDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=self.embedding_size, hidden_size=self.hidden_size,
                            num_layers=num_layers, dropout=self.dropout)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

    def forward(self, x, captions, temperature=0.1, max_length=20, gen=False):
        """ x is Encoder() output, captions is (batch_size, max_sentence_len) """
        self.lstm.flatten_parameters()
        x = torch.unsqueeze(x, dim=0)
        x = torch.cat((x, x), dim=0)
        hidden_state = (x, x)

        if gen:
            start_captions = torch.ones((x.size(1), 1)).long()
            output = torch.empty((x.size(1), max_length)).long()
            word_embed = self.embed(start_captions).permute(1, 0, 2)
            if temperature != 0:
                for t in range(0, max_length):
                    lstm_out, hidden_state = self.lstm(word_embed, hidden_state)
                    output[:, t] = torch.squeeze(torch.multinomial(F.softmax(self.fc_out(torch.squeeze(lstm_out, dim=0)) / temperature, dim=1), num_samples=1), dim=1)
                    word_embed = self.embed(torch.unsqueeze(output[:, t], dim=0))
            else:
                for t in range(0, max_length):
                    lstm_out, hidden_state = self.lstm(word_embed, hidden_state)
                    output[:, t] = torch.argmax(self.fc_out(torch.squeeze(lstm_out, dim=0)), dim=1)
                    word_embed = self.embed(torch.unsqueeze(output[:, t], dim=0))

            return output

        captions_embed = self.embed(captions).permute(1, 0, 2)
        lstm_out, _ = self.lstm(captions_embed, hidden_state)
        output = self.fc_out(lstm_out.permute(1, 0, 2))

        return output


class RNNDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size):
        super(RNNDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size

        self.rnn = nn.RNNCell(input_size=self.embedding_size, hidden_size=self.hidden_size)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

    def forward(self, x, captions, temperature=0.1, max_length=20, gen=False):
        """ x is Encoder() output, captions is (batch_size, max_sentence_len) """

        hidden_state = x
        if gen:
            start_captions = torch.ones((x.size(0), 1)).long()
            output = torch.empty((x.size(0), max_length)).long()
            word_embed = self.embed(start_captions)
            if temperature != 0:
                for t in range(0, max_length):
                    hidden_state = self.rnn(word_embed[:, 0, :], hidden_state)
                    output[:, t] = torch.squeeze(
                        torch.multinomial(F.softmax(self.fc_out(hidden_state) / temperature, dim=1), num_samples=1),
                        dim=1)
                    word_embed = self.embed(torch.unsqueeze(output[:, t], dim=1))
            else:
                for t in range(0, max_length):
                    hidden_state = self.rnn(word_embed[:, 0, :], hidden_state)
                    output[:, t] = torch.argmax(self.fc_out(hidden_state), dim=1)
                    word_embed = self.embed(torch.unsqueeze(output[:, t], dim=1))

            return output

        captions_embed = self.embed(captions)
        # shape (batch_size, max_sentence_length, embedding_dim)
        output = torch.empty((captions.size(0), captions.size(1), self.vocab_size))
        for t in range(captions.size(1)):
            hidden_state = self.rnn(captions_embed[:, t, :], hidden_state)

            output[:, t, :] = self.fc_out(hidden_state)

        return output


class StackedRNNDecoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, vocab_size, num_layers=2, dropout=0.0):
        super(StackedRNNDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.rnn = nn.RNN(input_size=self.embedding_size, hidden_size=self.hidden_size,
                          num_layers=num_layers, dropout=self.dropout)
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

    def forward(self, x, captions, temperature=0.1, max_length=20, gen=False):
        """ x is Encoder() output, captions is (batch_size, max_sentence_len) """
        self.rnn.flatten_parameters()
        x = torch.unsqueeze(x, dim=0)
        x = torch.cat((x, x), dim=0)
        hidden_state = x

        if gen:
            start_captions = torch.ones((x.size(1), 1)).long()
            output = torch.empty((x.size(1), max_length)).long()
            word_embed = self.embed(start_captions).permute(1, 0, 2)
            if temperature != 0:
                for t in range(0, max_length):
                    rnn_out, hidden_state = self.rnn(word_embed, hidden_state)
                    output[:, t] = torch.squeeze(torch.multinomial(F.softmax(self.fc_out(torch.squeeze(rnn_out, dim=0)) / temperature, dim=1), num_samples=1), dim=1)
                    word_embed = self.embed(torch.unsqueeze(output[:, t], dim=0))
            else:
                for t in range(0, max_length):
                    rnn_out, hidden_state = self.rnn(word_embed, hidden_state)
                    output[:, t] = torch.argmax(self.fc_out(torch.squeeze(rnn_out, dim=0)), dim=1)
                    word_embed = self.embed(torch.unsqueeze(output[:, t], dim=0))

            return output

        captions_embed = self.embed(captions).permute(1, 0, 2)
        rnn_out, _ = self.rnn(captions_embed, hidden_state)
        output = self.fc_out(rnn_out.permute(1, 0, 2))

        return output