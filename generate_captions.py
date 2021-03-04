
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import os
import time
import torchvision
from PIL import Image
import glob

import nltk
nltk.download('punkt')
from caption_utils import *
from dataset_factory import get_datasets
from file_utils import *
from model_factory_gen import get_model

ROOT_STATS_DIR = '.'
ROOT_CONFIGS_DIR = '.'


class GenExperiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', ROOT_CONFIGS_DIR + '/' + name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", ROOT_CONFIGS_DIR + '/' + name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        _, self.__vocab, _, _, _ = get_datasets(
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
        self.__best_loss = 1000000000000.
        self.__img_size = config_data['dataset']['img_size']
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__encoder = get_model(config_data, self.__vocab, network_block='encoder')
        if config_data['model']['model_type'] == 'LSTM':
            self.__decoder = get_model(config_data, self.__vocab, network_block='decoder')
        else:
            assert(0 == 1), 'must select valid model_type'

        self.__init_model()


    def __load_model(self):
        if os.path.exists('models/best_model.pt'):
            if torch.cuda.is_available():
                state_dict = torch.load('models/best_model.pt')
            else:
                state_dict = torch.load('models/best_model.pt', map_location=torch.device('cpu'))
            self.__encoder.load_state_dict(state_dict['encoder'])
            self.__decoder.load_state_dict(state_dict['decoder'])
        else:
            assert(0 == 1), 'model path doesn\'t exist'

    def __init_model(self):
        if torch.cuda.is_available():
            self.__encoder = nn.DataParallel(self.__encoder, device_ids=range(self.__num_gpu)).float().cuda()
            self.__decoder = nn.DataParallel(self.__decoder, device_ids=range(self.__num_gpu)).float().cuda()
        else:
            self.__encoder = nn.DataParallel(self.__encoder).float()
            self.__decoder = nn.DataParallel(self.__decoder).float()



    def gen_caption(self, temp=None, out_dir='./generated_output'):
        self.__load_model()
        self.__encoder.eval()
        self.__decoder.eval()
        img_path = self.__img_root_dir
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
                [torchvision.transforms.Resize(self.__img_size, interpolation=2), torchvision.transforms.CenterCrop(self.__img_size)])
            for img_num, img_fn in enumerate(glob.glob(img_path + '*')):
                image_original = plt.imread(img_fn)
                image = Image.open(img_fn).convert('RGB')
                image = resize_img(image)
                image = normalize_img(np.asarray(image))
                if torch.cuda.is_available():
                    X = image.cuda()
                else:
                    X = image
                img_embedded = self.__encoder(X.unsqueeze(0))
                gen_captions = self.__decoder(img_embedded, 0, gen=True, temperature=temperature, max_length=gen_max_length).type(torch.int).cpu()
                list_captions = [gen_captions[n, :] for n in range(gen_captions.size(0))]

                for n, gen_cap in enumerate(list_captions):
                    if 2 in gen_cap:
                        gen_cap = gen_cap[0:torch.where(gen_cap == 2)[0][0].item()]
                    if 1 in gen_cap:
                        gen_cap = gen_cap[torch.where(gen_cap == 1)[0][0].item()+1:]

                    parsed_gen_caption = ' '.join([self.__vocab.idx2word[index.item()] for index in gen_cap])
                    plt.figure(dpi=300)
                    plt.imshow(image_original)
                    plt.yticks([])
                    plt.xticks([])
                    plt.text(int(0.5 * image_original.shape[1]), image_original.shape[0] + 10, 'Generated Caption:\n' + parsed_gen_caption, wrap=True, ha='center', va='top', fontsize=10)
                    plt.subplots_adjust(bottom=0.30, top=0.99)
                    plt.savefig(out_dir + '/captioned_image_{}'.format(img_num), dpi=600, bbox_inches='tight',
                                pad_inches=0.0)
                    plt.close()

        else:
            assert(1 == 0), 'Must select image path'
