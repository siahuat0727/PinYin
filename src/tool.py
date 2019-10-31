
import argparse
from os import listdir
import re
from os.path import isfile, join

def parse():
  parser = argparse.ArgumentParser(description='PinYin input method',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--file1', type=str, default=None, help="predict result")
  parser.add_argument('--file2', type=str, default=None, help="correct answer")

  parser.add_argument('--compare', action='store_true')
  parser.add_argument('--compare-all', action='store_true')
  parser.add_argument('--dir', type=str, default='.', help="path to dir")
  parser.add_argument('--match', type=str, default='.*', help="regex to match files")
  args = parser.parse_args()
  return args

def compare(file1, file2):
  f1 = open(file1)
  f2 = open(file2)
  match = 0
  total = 0
  for line1, line2 in zip(f1, f2):
    for w1, w2 in zip(line1, line2):
      total += 1
      if w1 == w2:
        match += 1
  print(f'match: {match}, rate: {match/total}')
  f1.close()
  f2.close()

def main():
  args = parse()
  if args.compare:
    compare(args.file1, args.file2)
    return

  if args.compare_all:
    files = [
        join(args.dir, f)
        for f in listdir(args.dir)
        if isfile(join(args.dir, f)) and re.match(args.match, f) is not None
    ]
    for f in files:
      print(f)
      compare(f, args.file2)
    return


if __name__ == '__main__':
  main()
