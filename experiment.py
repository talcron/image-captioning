################################################################################
# CSE 253: Programming Assignment 4
# Code snippet by Ajit Kumar, Savyasachi
# Fall 2020
################################################################################

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import os
import time
import torchvision
from PIL import Image

import nltk
from caption_utils import *
from constants import ROOT_STATS_DIR, ROOT_CONFIGS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', ROOT_CONFIGS_DIR + '/' + name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", ROOT_CONFIGS_DIR + '/' + name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__img_root_dir = config_data['dataset']['images_root_dir']
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__lr = config_data['experiment']['learning_rate']
        self.__batch_size = config_data['dataset']['batch_size']
        self.__num_gpu = config_data['num_gpu']
        self.__vocab_size = len(self.__vocab)
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_loss = 1000000000000.
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__encoder = get_model(config_data, self.__vocab, network_block='encoder')
        if config_data['model']['model_type'] == 'LSTM':
            self.__decoder = get_model(config_data, self.__vocab, network_block='decoder')
        elif config_data['model']['model_type'] == 'stackedLSTM':
            self.__decoder = get_model(config_data, self.__vocab, network_block='stacked_decoder')
        elif config_data['model']['model_type'] == 'RNN':
            self.__decoder = get_model(config_data, self.__vocab, network_block='RNNdecoder')
        elif config_data['model']['model_type'] == 'stackedRNN':
            self.__decoder = get_model(config_data, self.__vocab, network_block='stacked_RNNdecoder')
        else:
            assert(0 == 1), 'must select valid model_type'

        # TODO: Set these Criterion and Optimizers Correctly
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(filter(
            lambda p: p.requires_grad, nn.ModuleList([self.__encoder, self.__decoder]
                                                     ).parameters()), lr=self.__lr)
        # If you use SparseAdam, change nn.Embedding in model_factory to sparse=True

        self.__init_model()

        # Load Experiment Data if available
        self.__load_experiment()

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__best_loss = read_file_in_dir(self.__experiment_dir, 'best_loss.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__encoder.load_state_dict(state_dict['encoder'])
            self.__decoder.load_state_dict(state_dict['decoder'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir, exist_ok=True)

    def __load_bleu(self):
        if os.path.exists(self.__experiment_dir + 'bleu1_scores.txt'):
            self.__bleu1 = read_file_in_dir(self.__experiment_dir, 'bleu1_scores.txt')
            self.__bleu4 = read_file_in_dir(self.__experiment_dir, 'bleu4_scores.txt')

    def __load_model(self, use_best=False):
        if os.path.exists(self.__experiment_dir):
            if use_best:
                state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model.pt'))
                self.__encoder.load_state_dict(state_dict['encoder'])
                self.__decoder.load_state_dict(state_dict['decoder'])
                self.__optimizer.load_state_dict(state_dict['optimizer'])
            else:
                state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
                self.__encoder.load_state_dict(state_dict['encoder'])
                self.__decoder.load_state_dict(state_dict['decoder'])
                self.__optimizer.load_state_dict(state_dict['optimizer'])
        else:
            assert(0 == 1), 'model path doesn\'t exist'

    def __init_model(self):
        if torch.cuda.is_available():
            # self.__model = self.__model.cuda().float()
            self.__encoder = nn.DataParallel(self.__encoder, device_ids=range(self.__num_gpu)).float().cuda()
            self.__decoder = nn.DataParallel(self.__decoder, device_ids=range(self.__num_gpu)).float().cuda()
            self.__criterion = self.__criterion.cuda()


    # Main method to run your experiment. Should be self-explanatory.
    def run(self, get_bleu=False):
        if get_bleu:
            self.__bleu1 = []
            self.__bleu4 = []
            self.__load_bleu()
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            print('\nFinished Training Epoch {}'.format(epoch+1))
            val_loss = self.__val()
            print('\nFinished Validation Epoch {}'.format(epoch+1))
            is_best = self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if is_best:
                self.__save_best_model()
            if get_bleu:
                b1, b4 = self.__bleu_test()
                self.__record_bleu(b1, b4)



    # TODO: Perform one training iteration on the whole dataset and return loss value
    def __train(self):
        # train_loader returns (imgs, labels, img_ids), imgs is just
        #images (batch_size, 3, 256, 256), labels is (batch_size, max_sentence_length_for_this_batch)
        # labels consists of integers that map to a unique word in vocabulary
        #1 is start, 2 is end, 0 is padding
        self.__encoder.train()
        self.__decoder.train()
        train_cost = torch.zeros(1).cuda()
        for i, (images, captions, _) in enumerate(self.__train_loader):

            X = images.cuda()
            Y_target = captions[:, 1:].cuda()
            Y_train = captions[:, :captions.size(1) - 1].cuda()

            img_embedded = self.__encoder(X)
            out_captions = self.__decoder(img_embedded, Y_train)
            loss = self.__criterion(out_captions.view(-1, self.__vocab_size), Y_target.contiguous().view(-1))
            train_cost += loss.detach()


            self.__optimizer.zero_grad()
            loss.backward()
            self.__optimizer.step()

            # raise NotImplementedError()
        return (train_cost.item()) / len(self.__train_loader)

    # TODO: Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__encoder.eval()
        self.__decoder.eval()
        val_cost = torch.zeros(1).cuda()

        with torch.no_grad():
            for i, (images, captions, _) in enumerate(self.__val_loader):
                X = images.cuda()
                Y_target = captions[:, 1:].cuda()
                Y_train = captions[:, :captions.size(1) - 1].cuda()

                img_embedded = self.__encoder(X)
                out_captions = self.__decoder(img_embedded, Y_train)
                loss = self.__criterion(out_captions.view(-1, self.__vocab_size), Y_target.contiguous().view(-1))
                val_cost += loss.detach()

                # raise NotImplementedError()

        return (val_cost.item()) / len(self.__val_loader)

    # TODO: Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self, use_best=False, temp=None):
        if use_best:
            self.__load_model(use_best=True)
        else:
            self.__load_model()
        self.__encoder.eval()
        self.__decoder.eval()
        test_cost = torch.zeros(1).cuda()
        bleu1_score = 0
        bleu4_score = 0
        if temp is None:
            temperature = self.__generation_config['temperature']
        else:
            temperature = temp
        gen_max_length = self.__generation_config['max_length']

        # total_iter = len(self.__test_loader)
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                # print('Iteration {} of {}'.format(iter, total_iter))
                #img_ids is a list of integers
                X = images.cuda()
                Y_target = captions[:, 1:].cuda()
                Y_train = captions[:, :captions.size(1) - 1].cuda()
                img_embedded = self.__encoder(X)

                out_captions = self.__decoder(img_embedded, Y_train)
                loss = self.__criterion(out_captions.view(-1, self.__vocab_size), Y_target.contiguous().view(-1))
                test_cost += loss.detach()
                gen_captions = self.__decoder(img_embedded, Y_train, gen=True, temperature=temperature, max_length=gen_max_length).type(torch.int).cpu()
                # end_idx = [torch.where(gen_captions[n, :] == 2)[1] for n in range(gen_captions.size(0))]
                list_captions = [gen_captions[n, :] for n in range(gen_captions.size(0))]

                for n, gen_cap in enumerate(list_captions):
                    if 2 in gen_cap:
                        gen_cap = gen_cap[0:torch.where(gen_cap == 2)[0][0].item()]
                    if 1 in gen_cap:
                        gen_cap = gen_cap[torch.where(gen_cap == 1)[0][0].item()+1:]

                    caption_dict = self.__coco_test.imgToAnns[img_ids[n]]
                    # real_captions = [entry['caption'].lower().split() for entry in caption_dict]
                    real_captions = [nltk.tokenize.word_tokenize(entry['caption'].lower()) for entry in caption_dict]
                    parsed_gen_caption = [self.__vocab.idx2word[index.item()] for index in gen_cap]
                    bleu1_score += bleu1(real_captions, parsed_gen_caption)
                    bleu4_score += bleu4(real_captions, parsed_gen_caption)
        if use_best:
            result_str = "Test Performance (Best Model): Loss: {}, Bleu1: {}, Bleu4: {}, Temp: {}"\
                .format((test_cost.item() / len(self.__test_loader)),
                   bleu1_score / (len(self.__test_loader) * self.__batch_size),
                   bleu4_score / (len(self.__test_loader) * self.__batch_size),
                        temperature)
        else:
            result_str = "Test Performance (Latest Model): Loss: {}, Bleu1: {}, Bleu4: {}, Temp {}"\
                .format((test_cost.item() / len(self.__test_loader)),
                   bleu1_score / (len(self.__test_loader) * self.__batch_size),
                   bleu4_score / (len(self.__test_loader) * self.__batch_size),
                        temperature)
        self.__log(result_str)
        return (test_cost.item() / len(self.__test_loader)),\
               bleu1_score / (len(self.__test_loader) * self.__batch_size),\
               bleu4_score / (len(self.__test_loader) * self.__batch_size)

    def __bleu_test(self, temp=None):
        self.__encoder.eval()
        self.__decoder.eval()
        bleu1_score = 0
        bleu4_score = 0
        if temp is None:
            temperature = self.__generation_config['temperature']
        else:
            temperature = temp
        gen_max_length = self.__generation_config['max_length']

        # total_iter = len(self.__test_loader)
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                # print('Iteration {} of {}'.format(iter, total_iter))
                #img_ids is a list of integers
                X = images.cuda()
                Y_train = captions[:, :captions.size(1) - 1].cuda()
                img_embedded = self.__encoder(X)

                gen_captions = self.__decoder(img_embedded, Y_train, gen=True, temperature=temperature, max_length=gen_max_length).type(torch.int).cpu()
                # end_idx = [torch.where(gen_captions[n, :] == 2)[1] for n in range(gen_captions.size(0))]
                list_captions = [gen_captions[n, :] for n in range(gen_captions.size(0))]
                # print(list_captions[0])
                # print(list_captions[1])
                # print(list_captions[2])


                for n, gen_cap in enumerate(list_captions):
                    if 2 in gen_cap:
                        gen_cap = gen_cap[0:torch.where(gen_cap == 2)[0][0].item()]
                    if 1 in gen_cap:
                        gen_cap = gen_cap[torch.where(gen_cap == 1)[0][0].item()+1:]

                    caption_dict = self.__coco_test.imgToAnns[img_ids[n]]
                    # real_captions = [entry['caption'].lower().split() for entry in caption_dict]
                    real_captions = [nltk.tokenize.word_tokenize(entry['caption'].lower()) for entry in caption_dict]
                    parsed_gen_caption = [self.__vocab.idx2word[index.item()] for index in gen_cap]
                    bleu1_score += bleu1(real_captions, parsed_gen_caption)
                    bleu4_score += bleu4(real_captions, parsed_gen_caption)

        result_str = "Test BLEU scores (Latest Model): Bleu1: {}, Bleu4: {}, Temp {}"\
            .format(bleu1_score / 15000.,
                    bleu4_score / 15000.,
                    temperature)
        self.__log(result_str)
        return bleu1_score / 15000.,\
               bleu4_score / 15000.


    def gen_caption(self, temp=None, img_path=None, out_dir='./outputs'):
        self.__load_model(use_best=True)
        self.__encoder.eval()
        self.__decoder.eval()
        os.makedirs(out_dir, exist_ok=True)
        if temp is None:
            temperature = self.__generation_config['temperature']
        else:
            temperature = temp
        gen_max_length = self.__generation_config['max_length']
        if img_path is not None:
            assert(os.path.exists(img_path)), 'img_path does not exists'
            normalize_img = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            resize_img = torchvision.transforms.Compose(
                [torchvision.transforms.Resize(256, interpolation=2), torchvision.transforms.CenterCrop(256)])
            image = Image.open(img_path).convert('RGB')
            image = resize_img(image)
            image = normalize_img(np.asarray(image))
        else:
            root_test = os.path.join(self.__img_root_dir, 'test')
            with torch.no_grad():
                images, captions, img_ids = next(iter(self.__test_loader))
                X = images.cuda()
                Y_train = captions[:, :captions.size(1) - 1].cuda()
                img_embedded = self.__encoder(X)

                gen_captions = self.__decoder(img_embedded, Y_train, gen=True, temperature=temperature, max_length=gen_max_length).type(torch.int).cpu()
                list_captions = [gen_captions[n, :] for n in range(gen_captions.size(0))]

                for n, gen_cap in enumerate(list_captions):
                    if 2 in gen_cap:
                        gen_cap = gen_cap[0:torch.where(gen_cap == 2)[0][0].item()]
                    if 1 in gen_cap:
                        gen_cap = gen_cap[torch.where(gen_cap == 1)[0][0].item()+1:]

                    caption_dict = self.__coco_test.imgToAnns[img_ids[n]]
                    real_captions = [entry['caption'].lower() for entry in caption_dict]
                    parsed_gen_caption = ' '.join([self.__vocab.idx2word[index.item()] for index in gen_cap])
                    path = self.__coco_test.loadImgs(img_ids[n])[0]['file_name']
                    image = Image.open(os.path.join(root_test, path))
                    image.save(out_dir + '/captioned_{}_image.png'.format(n))
                    with open(out_dir + '/captioned_{}_real.txt'.format(n), "w") as out_file:
                        for line in real_captions:
                            out_file.write(line)
                    with open(out_dir + '/captioned_{}_gen.txt'.format(n), "w") as out_file:
                        out_file.write(parsed_gen_caption)



    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        encoder_dict = self.__encoder.state_dict()
        decoder_dict = self.__decoder.state_dict()
        state_dict = {'encoder': encoder_dict, 'decoder': decoder_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __save_best_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'best_model.pt')
        encoder_dict = self.__encoder.state_dict()
        decoder_dict = self.__decoder.state_dict()
        state_dict = {'encoder': encoder_dict, 'decoder': decoder_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        best_flag = False
        if val_loss < self.__best_loss:
            self.__best_loss = val_loss
            write_to_file_in_dir(self.__experiment_dir, 'best_loss.txt', self.__best_loss)
            best_flag = True

        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        # self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)
        return best_flag

    def __record_bleu(self, bleu1_score, bleu4_score):

        self.__bleu1.append(bleu1_score)
        self.__bleu4.append(bleu4_score)

        write_to_file_in_dir(self.__experiment_dir, 'bleu1_scores.txt', self.__bleu1)
        write_to_file_in_dir(self.__experiment_dir, 'bleu4_scores.txt', self.__bleu4)


    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()
