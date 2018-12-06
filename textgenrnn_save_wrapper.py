import shutil, os
import argparse
from textgenrnn import textgenrnn


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", required=True,
                        help="name of model & folder where it will be saved")
    parser.add_argument("--new_model", default=False,
                        help="Include to start training from scratch. Otherwise, will try to load a model from "
                             "weights/model_name",
                        action="store_true")
    parser.add_argument("--num_epochs", type=int, default=0,
                        help="number of times training process will go through the entire training dataset. "
                             "if 0, just generates samples.")
    parser.add_argument("--data_file", help="Location of training data file. Assumed to be in data folder")
    parser.add_argument("--large_text", default=False,
                        help="Include to train in longtext mode instead of treating each line as a new item",
                        action="store_true")
    parser.add_argument("--random_batch", type=int, default=0,
                        help="Randomly choose a batch from the input text and train.")
    parser.add_argument("--word_level", default=False,
                        help="Include to train in word mode rather than character mode", action="store_true")
    parser.add_argument("--max_words", type=int, default=20000,
                        help="maximum number of words to include in vocab for word mode")
    parser.add_argument("--save_name", default='',
                        help="Include to save the model in a new location when done. Otherwise, will save in "
                             "weights/model_name")
    parser.add_argument("--n_gen", type=int, default=1,
                        help="number of output samples to generate")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="temperature to use when generating samples")
    parser.add_argument("--prefix", default='',
                        help="Starting text to use for sample generation")
    parser.add_argument("--max_gen_length", type=int, default=1000,
                        help="For sampling large_text models, the max length of a single sample")
    return parser


def load_model(load_loc):
    return textgenrnn(
        weights_path=os.path.join(load_loc, 'textgenrnn_weights.hdf5'),
        vocab_path=os.path.join(load_loc, 'textgenrnn_vocab.json'),
        config_path=os.path.join(load_loc, 'textgenrnn_config.json')
    )
    
    
def random_choose(texts, size):
    import random
    random.shuffle(texts)
    return texts[:size]


if __name__ == '__main__':

    if not os.path.exists('models'):
        os.mkdir('models')
    if not os.path.exists('data'):
        os.mkdir('data')

    parser = args_parser()
    args = parser.parse_args()

    load_loc = os.path.join('models', args.model_name)
    data_loc = os.path.join('data', args.data_file or '')
    save_loc = os.path.join('models', args.save_name or args.model_name)

    if args.num_epochs == 0:  # sampling
        my_model = load_model(load_loc)
        my_model.generate(n=args.n_gen,
                          temperature=args.temperature,
                          prefix=args.prefix,
                          max_gen_length=args.max_gen_length)

    else:
        try:
            args.new_model = args.new_model or not os.path.exists(load_loc)
            my_model = textgenrnn() if args.new_model else load_model(load_loc)

            if args.word_level:
                print('Using word-level mode.')
                my_model.train_from_file(data_loc, num_epochs=args.num_epochs, new_model=args.new_model, word_level=True,
                                         max_words=args.max_words)
            elif args.large_text:
                print('Using large-text mode.')
                my_model.train_from_largetext_file(data_loc, num_epochs=args.num_epochs, new_model=args.new_model)
            elif args.random_batch:
                print('Using random batch mode.')
                texts = open(data_loc, 'r', encoding='utf-8').readlines()
                for i in range(args.num_epochs):
                    my_model.train_on_texts(texts=random_choose(texts, args.random_batch), num_epochs=1, new_model=args.new_model)
            else:
                print('Using normal text mode.')
                my_model.train_from_file(data_loc, num_epochs=args.num_epochs, new_model=args.new_model)
        except KeyboardInterrupt:
            pass

        if not os.path.exists(save_loc):
            os.makedirs(save_loc)
        for f in ['textgenrnn_weights.hdf5', 'textgenrnn_vocab.json', 'textgenrnn_config.json']:
            shutil.copy(f, save_loc)
