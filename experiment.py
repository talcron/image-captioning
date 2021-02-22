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

import nltk
from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
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
        self.__decoder = get_model(config_data, self.__vocab, network_block='decoder')

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

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'best_model.pt'))
            self.__encoder.load_state_dict(state_dict['encoder'])
            self.__decoder.load_state_dict(state_dict['decoder'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir, exist_ok=True)

    def __init_model(self):
        if torch.cuda.is_available():
            # self.__model = self.__model.cuda().float()
            self.__encoder = nn.DataParallel(self.__encoder, device_ids=range(self.__num_gpu)).float().cuda()
            self.__decoder = nn.DataParallel(self.__decoder, device_ids=range(self.__num_gpu)).float().cuda()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            print('\nFinished Training Epoch {}\nLoss: {:.4f}'.format(epoch+1, train_loss))
            val_loss = self.__val()
            print('\nFinished Validation Epoch {}\nLoss: {:.4f}'.format(epoch+1, val_loss))
            is_best = self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()
            if is_best:
                self.__save_best_model()

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
    def test(self):
        self.__encoder.eval()
        self.__decoder.eval()
        test_cost = torch.zeros(1).cuda()
        bleu1_score = 0
        bleu4_score = 0
        temperature = self.__generation_config['temperature']
        gen_max_length = self.__generation_config['max_length']

        total_iter = len(self.__test_loader)
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                print('Iteration {} of {}'.format(iter, total_iter))
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


        result_str = "Test Performance: Loss: {}, Bleu1: {}, Bleu4: {}"\
            .format((test_cost.item() / len(self.__test_loader)),
               bleu1_score / (len(self.__test_loader) * self.__batch_size),
               bleu4_score / (len(self.__test_loader) * self.__batch_size))
        self.__log(result_str)
        return (test_cost.item() / len(self.__test_loader)),\
               bleu1_score / (len(self.__test_loader) * self.__batch_size),\
               bleu4_score / (len(self.__test_loader) * self.__batch_size)


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
