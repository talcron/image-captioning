# Image Captioning

## Usage

* Create a directory named 'models' in code directory
* Download best_model.pt and place into models directory
* best_model.pt specifies parameters for a single-cell LSTM-based RNN decoder with embedding and output FC layers to transform word tokens
* place any images you want captions to be generated for in the img_data/images/ directory
* outputs will be saved to generated_output/
* for caption generation, simply run 'python main_generate.py'

## Files
- main.py: Main driver class
- generate_main.py: just runs image generation using pretrained network best_model.pt
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- generate_captions.py: stripped down version of experiment.py, only contains code for generation using pretrained model
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
