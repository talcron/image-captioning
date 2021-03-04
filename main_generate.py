
from generate_captions import GenExperiment
import sys

# Main Driver for your code. Either run `python main.py` which will run the experiment with default config
if __name__ == "__main__":
    exp_name = 'default'

    exp = GenExperiment(exp_name)
    exp.gen_caption()
