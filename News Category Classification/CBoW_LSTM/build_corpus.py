import re
import sys
import argparse
import pandas as pd


def argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--corpus', default=None, type=str, required=True)

    config = p.parse_args()
    return config


if __name__ == '__main__':
    config = argparser()

    data = pd.read_json(config.corpus, lines=True)

    data = data.loc[:, ['category', 'headline', 'short_description']]

    corpus = data['headline'].str.strip() + '. ' + data['short_description'].str.strip()
    labels = data['category'].str.strip()
    lines = ''
    for i, (text, label) in enumerate(zip(corpus, labels)):
        print(i)
        lines += '{}\t{}'.format(label, re.sub(r'\n', ' ', text))+r'\n'

        #sys.stdout.write(line + '\n')

    with open('corpus/corpus.txt', 'a+') as file:
        file.write(lines)
    file.close()
