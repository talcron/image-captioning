################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import torch
import torch.nn as nn
import torchvision


# Build and return the model here based on the configuration.
def get_model(config_data, vocab):
    """ vocab is Vocabulary() object from vocab.py """
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']

    encoder = Encoder(hidden_size)
    decoder = Decoder(hidden_size=hidden_size, embedding_size=embedding_size, vocab_size=len(vocab))


    raise NotImplementedError("Model Factory Not Implemented")



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
        self.fc_out = nn.Linear(in_features=hidden_size, out_features=self.vocab_size)
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)

    def forward(self, x, captions):
        """ x is Encoder() output """
        batch_size = x.size(0)

        hidden_state = x








# class Resnet(nn.Module):
#     def __init__(self, pretrained_model):
#         super(Resnet, self).__init__()
#
#         self.relu = nn.ReLU(inplace=True)
#
#         self.pretrained = pretrained_model.features
#         self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1,
#                                           bias=False)
#         self.bn1 = nn.BatchNorm2d(512)
#         self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1,
#                                           bias=False)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1,
#                                           bias=False)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1,
#                                           bias=False)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1,
#                                           bias=False)
#         self.bn5 = nn.BatchNorm2d(32)
#         self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
#
#     def forward(self, x):
#         out_encoder = self.pretrained(x)
#         y = self.bn1(self.relu(self.deconv1(out_encoder)))
#         y = self.bn2(self.relu(self.deconv2(y)))
#         y = self.bn3(self.relu(self.deconv3(y)))
#         y = self.bn4(self.relu(self.deconv4(y)))
#         out_decoder = self.bn5(self.relu(self.deconv5(y)))
#
#         score = self.classifier(out_decoder)
#         return score

# def init_transfer_weights(m):
#     if isinstance(m, nn.ConvTranspose2d):
#         torch.nn.init.xavier_uniform_(m.weight.data)
#         if m.bias is not None:
#             torch.nn.init.zeros_(m.bias.data)
