# -*- coding: utf-8 -*-

import re
import sys
import json
import pickle
import argparse
import itertools
from time import time
from os import listdir
from os.path import isfile, join
from functools import partial
from collections import defaultdict

def parse():
  parser = argparse.ArgumentParser(description='PinYin Input Method',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # IOs
  parser.add_argument('--verbose', action='store_true', help="whether to print more information")
  parser.add_argument('--input-file', type=str, default=None, help="input file (if any)")
  parser.add_argument('--output-file', type=str, default=None, help="output file (if any)")

  # Paths
  parser.add_argument('--load-model', type=str, default=None, help="path to load model")
  parser.add_argument('--save-model', type=str, default='../model/model.pkl', help="path to save model")
  parser.add_argument('--words', type=str, default='../model/words.pkl', help="path to save words")
  parser.add_argument('--pinyin-table', type=str, default='../model/pinyin_table.pkl', help="path to save pinyin-table")

  # Tasks
  parser.add_argument('--init-words', action='store_true', help="task: init words")
  parser.add_argument('--init-pinyin-table', action='store_true', help="task: init pinyin-table")
  parser.add_argument('--train', action='store_true', help="task: train model")
  parser.add_argument('--analysis', action='store_true', help="task: analysis model")

  # Files
  parser.add_argument('--encoding', type=str, default='utf8', help="input file coding method")
  parser.add_argument('--file', type=str, default=None, help="path to file")
  parser.add_argument('--dir', type=str, default=None, help="path to dir")
  parser.add_argument('--match', type=str, default='.*', help="regex to match training files when given directory")

  # Model params
  parser.add_argument('--alpha', type=float, default=0.9, help="smoothing factor")
  parser.add_argument('--n-gram', type=int, default=2, help="using n-gram model")
  parser.add_argument('--no-clip', action='store_true', help="disable clip for words (check README.md for detail)")

  # Methods
  parser.add_argument('--brute-force', action='store_true', help="use brute force (instead of dynamic programming)")
  parser.add_argument('--fast', action='store_true', help="only find approx. answer when using dynamic programming")

  # Trick
  parser.add_argument('--slim', action='store_true', help="make pinyin-table slimmer")
  parser.add_argument('--threshold', type=int, default=100,
                      help="del a word from pinyin-table if # of the word is less than threshold")

  args = parser.parse_args()
  print(args)
  assert 0.0 <= args.alpha <= 1.0
  assert args.n_gram >= 2
  return args

def return_param(param):
  '''Return anything (since pickle can't save anything with lambda function)'''
  return param

def n_defaultdict(default_param, n=1):
  '''Create dictionary of dictionary of dict... *n with default value default_param'''
  my_dict = partial(return_param, default_param)
  for _ in range(n):
    my_dict = partial(defaultdict, my_dict)
  return my_dict()

def load_file(path, encoding, verbose):
  if verbose:
    print(f'Loading {path}', end='')
  with open(path, 'r', encoding=encoding) as f:
    lines = f.readlines()
  if verbose:
    print(f'\rLoaded {path} ')
  return lines

def load_pickle(path, verbose):
  if verbose:
    print(f'Loading {path}', end='')
  with open(path, 'rb') as f:
    data = pickle.load(f)
  if verbose:
    print(f'\rLoaded {path} ')
  return data

def save_pickle(data, path, verbose=True):
  if verbose:
    print(f'Saving {path}', end='')
  with open(path, 'wb') as f:
    pickle.dump(data, f)
  if verbose:
    print(f'\rSaved {path} ')

def init_words(args):
  '''Create a dictionary to check whether a word exist'''
  words_exist = n_defaultdict(False)
  words = load_file(args.file, args.encoding, args.verbose)[0]
  for word in words:
    words_exist[word] = True
  save_pickle(words_exist, args.words)

def init_pinyin_table(args):
  '''Create the mapping from pinyin to words'''
  pinyin2words = n_defaultdict(['@'])  # For @-clipping magic

  lines = load_file(args.file, args.encoding, args.verbose)
  for line in lines:
    pinyin, *words = line.strip().split(' ')  # e.g. line = "a 啊 嗄 腌 吖 阿 锕"
    pinyin2words[pinyin] = words

  save_pickle(pinyin2words, args.pinyin_table)

def get_clean_text(line, words_exist):
  data = json.loads(line)
  # Get text (with unknown words)
  words = '@'.join((data['html'], data['title']))
  # Replace all unknown words to '@'
  words = ''.join(
      word if words_exist[word] else '@'
      for word in words
  )
  # Add '@' to both head and end of the string
  words = f'@{words}@'  # this looks so cute!
  # Remove duplicate '@'
  words = re.sub(r'(@)\1+', r'\1', words)
  return words

def get_clean_key(key):
  '''For n-gram with n>2 (i.e. len(key)>=2), anything before '@' is useless. Ex. 'x@y'->'y' '''
  if key[-1] == '@':
    return key[-1]
  return key.split('@')[-1]

def train(args, model, words_exist):
  def do_train(path):

    def do_record(prefix, word):
      model[prefix][word] += 1

      if len(prefix) > 1:
        model[prefix]['total'] += 1  # Already recorded for len = 1
        do_record(prefix[1:], word)


    print(f'Start training for {path}')
    with open(path, 'r', encoding=args.encoding) as f:
      for idx, line in enumerate(f):
        words = get_clean_text(line, words_exist)

        # Record for smoothing
        words_length = len(words)
        for word in words:
          model[word]['total'] += 1
        model['all']['total'] += words_length

        n_grams = (
            words[s:s+args.n_gram]
            for s in range(words_length - args.n_gram + 1)
        )
        for n_gram in n_grams:
          prefix, word = n_gram[:-1], n_gram[-1]
          prefix = get_clean_key(prefix)
          do_record(prefix, word)
        print(f'\r{idx+1} news trained', end='')
      print()
      return

  assert args.dir is not None or args.file is not None, \
      "Either file or dir should be given for training data"

  if args.file is not None:
    do_train(args.file)
    return

  files = [
      join(args.dir, f)
      for f in listdir(args.dir)
      if isfile(join(args.dir, f)) and re.match(args.match, f) is not None
  ]

  if args.verbose:
    print(f"Training files are:")
    for f in files:
      print(f'\t{f}')

  for f in files:
    do_train(f)

def solve(args, model, pinyin2words, instream=sys.stdin, outstream=sys.stdout):
  def calc_prob(prefix, word):
    '''Calculate P#(word | prefix[-1:]) + P#(word | prefix[-2:]) + ... + P#(word | prefix)'''

    def do_calc(prefix, word):
      '''Calculate P#(word | prefix)'''
      eps = 1e-8
      continuous = 0
      if prefix in model:  # Avoid creating too many empty dict
        continuous = model[prefix].get(word, 0) / (model[prefix]['total'] + eps)
      alone = model[word]['total'] / model['all']['total']
      return args.alpha*continuous + (1-args.alpha)*alone

    key_length = args.n_gram - 1
    prefix = get_clean_key(prefix[-key_length:])
    probs = (do_calc(prefix[-i:], word) for i in range(1, len(prefix)+1))
    return sum(probs)*100  # *100 -> Avoid floating point unferflow when processing very long sentence

  def find_max_pair(pairs):
    '''Find the pair with maximum probability'''
    def prob_of_pair(pair):
      (text, prob) = pair
      return prob

    return max(pairs, key=prob_of_pair)

  def dfs(words_list):
    '''Depth-first search (brute force)'''

    def do_dfs(nexts, text, prob):
      if not nexts:
        return text, prob
      t_p_pairs = (
          do_dfs(nexts[1:], text+word, prob*calc_prob(text, word))
          for word in nexts[0]
      )
      return find_max_pair(t_p_pairs)

    t_p_pairs = (
        do_dfs(words_list[1:], word, model[word]['total'])
        for word in words_list[0]
    )
    text, prob = find_max_pair(t_p_pairs)
    return text

  def dp(words_list):
    '''Dynamic programing (efficient)'''

    def do_dp_fast(words_list):
      '''Not perfect for n-gram model with n > 2 but faster. Return a list of tuple(text, prob)'''

      if len(words_list) == 1:
        return [(word, model[word]['total']) for word in words_list[0]]

      *prev_words_list, last_words = words_list

      prev_t_p_pairs = do_dp_fast(prev_words_list)

      t_p_pairs_unique_end = []
      for word in last_words:
        t_p_pairs = (
            (text+word, prob*calc_prob(text, word))
            for text, prob in prev_t_p_pairs
        )
        t_p_pairs_unique_end.append(find_max_pair(t_p_pairs))
      return t_p_pairs_unique_end

    def do_dp(words_list):
      '''Perfect for all n-gram model (always the same with dfs). Return a list of tuple(text, prob)'''

      def permutes_of_last_words(words_list):
        ''' Return a generator of generators ...
            Similar to itertools.product but:
            1. different order (sorted by ending character)
            2. different dimension (extra dimention grouping by ending character)
            All these difference are meaningful!
            Input : [['a', 'b'], ['c', 'd'], ['e', 'f']]
            Output: [[('a', 'c', 'e'), ('a', 'd', 'e'), ('b', 'c', 'e'), ('b', 'd', 'e')],
                     [('a', 'c', 'f'), ('a', 'd', 'f'), ('b', 'c', 'f'), ('b', 'd', 'f')]]
        '''
        *prev_words_list, last_words = words_list
        return (
            (permute + (last_word, ) for permute in itertools.product(*prev_words_list))
            for last_word in last_words
        )

      def calc_prob_multi(words, text, prob):
        if text == "":
          text, words = words[0], words[1:]
          prob = model[text]['total']

        for word in words:
          prob *= calc_prob(text, word)
          text += word
        return prob

      key_length = args.n_gram - 1
      prev_words_list, end_words_list = words_list[:-key_length], words_list[-key_length:]

      if len(words_list) <= key_length:
        prev_t_p_pairs = [("", None)]
      else:
        prev_t_p_pairs = do_dp(prev_words_list)

      t_p_pairs_unique_end = []
      for words_list_same_end in permutes_of_last_words(end_words_list):
        t_p_pairs = (
            (text + "".join(words), calc_prob_multi(words, text, prob))
            for (text, prob), words in itertools.product(prev_t_p_pairs, words_list_same_end)
        )
        t_p_pairs_unique_end.append(find_max_pair(t_p_pairs))
      return t_p_pairs_unique_end


    do_dp_ = do_dp_fast if args.fast else do_dp
    t_p_pairs = do_dp_(words_list)
    text, prob = find_max_pair(t_p_pairs)
    return text


  for line in instream:
    start = time()

    pinyins = line.strip().lower()
    if not args.no_clip:
      pinyins = f'@ {pinyins} @'
    pinyin_list = pinyins.split(' ')
    words_list = [pinyin2words[pinyin] for pinyin in pinyin_list]

    do_solve = dfs if args.brute_force else dp
    text = do_solve(words_list)
    if not args.no_clip:
      text = text[1:-1]  # remove both clipping-'@'

    end = time()

    if args.verbose:
      print(f'{pinyins} -> {text}, using {end - start:.3e} s')
    print(f'{text}', file=outstream)

def analysis(model):
  print('Input 2 keys, output the number of times key1 followed by key2.')
  print('For example,')
  print('Input: 你 好')
  print('Output: 1234 (the number of times 你 followed by 好)')
  print('-'*50)
  while True:
    try:
      prefix, word = input().strip().split()
      print(model[prefix][word])
    except Exception as e:
      print(e)
      break

def slim(pinyin2words, model, threshold):
  '''Remove low frequent words'''
  for pinyin in list(pinyin2words.keys()):
    pinyin2words[pinyin] = [
        word
        for word in pinyin2words[pinyin]
        if model[word]['total'] >= threshold
    ]
    if not pinyin2words[pinyin]:
      del pinyin2words[pinyin]

def main():
  args = parse()

  if args.init_words:
    init_words(args)
    return

  if args.init_pinyin_table:
    init_pinyin_table(args)
    return

  if args.load_model is None:
    model = n_defaultdict(0, n=2)
  else:
    model = load_pickle(args.load_model, args.verbose)

  if args.analysis:
    analysis(model)
    return

  if args.train:
    words_exist = load_pickle(args.words, args.verbose)
    train(args, model, words_exist)
    save_pickle(model, args.save_model)
    return

  pinyin2words = load_pickle(args.pinyin_table, args.verbose)
  if args.slim:
    slim(pinyin2words, model, args.threshold)

  iostreams = {}
  if args.input_file is not None:
    iostreams['instream'] = open(args.input_file, 'r', encoding=args.encoding)
  if args.output_file is not None:
    iostreams['outstream'] = open(args.output_file, 'w')

  solve(args, model, pinyin2words, **iostreams)

  # Release file resources if any
  for _, f in iostreams.items():
    f.close()
  return

if __name__ == '__main__':
  main()
