"""
@author:    Patrik Purgai
@copyright: Copyright 2019, supervised-translation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

import torch
import requests
import shutil
import os
import random
import json

from tqdm import tqdm
from itertools import (
    zip_longest, chain)

from os.path import (
    exists, join,
    dirname, abspath,
    basename, splitext)

from torch.utils.data import (
    Dataset, DataLoader,
    Sampler)

from torch.utils.data.distributed import (
    DistributedSampler)


START = '<sos>'
END = '<eos>'
PAD = '<pad>'
UNK = '<unk>'


def setup_data_args(parser):
    """
    Sets up the data arguments.
    """
    parser.add_argument(
        '--data_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the data root directory.')
    parser.add_argument(
        '--download_dir',
        type=str,
        default=join(abspath(dirname(__file__)), '..', 'data'),
        help='Path of the download directory.')
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=20000,
        help='Maximum size of the vocabulary.')
    parser.add_argument(
        '--min_freq',
        type=int,
        default=1,
        help='Minimum frequency of a word in the vocab.')
    parser.add_argument(
        '--max_len',
        type=int,
        default=50,
        help='Maximum length of a sequence.')


def download(args):
    """
    Downloads and extracts the daily dialog dataset from 
    google drive.
    """
    base_url = 'https://drive.google.com/uc?export=download&' + \
               'id=0B_bZck-ksdkpM25jRUN2X2UxMm8'
    filename = 'wmt16_en_de.tar.gz'

    if not exists(args.download_dir):
        os.mkdir(args.download_dir)

    if not exists(args.data_dir):
        os.mkdir(args.data_dir)

    url = base_url + filename
    download_path = join(args.download_dir, filename)

    if not exists(download_path):
        print('Downloading dataset to {}'.format(
            download_path))
        with requests.Session() as session:
            response = session.get(
                url, stream=True, timeout=5)

            with open(download_path, 'wb') as f:
                for chunk in tqdm(response.iter_content(
                        2 ** 15)):
                    if chunk:
                        f.write(chunk)

    train_path = join(args.data_dir, 'train.json')
    valid_path = join(args.data_dir, 'valid.json')
    test_path = join(args.data_dir, 'test.json')

    if not exists(train_path) or not \
            exists(valid_path) or not exists(test_path):
        print('Extracting dataset to {}'.format(
            args.data_dir))
        shutil.unpack_archive(download_path, args.data_dir)


def transform(args, tokenizer):
    """
    Transforms the dataset splits and smaller
    binary datafiles.
    """
    print('Transforming dataset')

    splits = [
        ('train.txt', 0.95),
        ('valid.txt', 0.025),
        ('test.txt', 0.025)
    ]

    train, valid, test = [
        (save_examples(
            args=args,
            data_path=filename,
            tokenizer=tokenizer), size)
        for filename, size in
        generate_splits(args, splits)]

    return train, valid, test


def compute_lines(filename):
    """
    Computes the number of lines of a file.
    """
    with open(filename, 'r') as fh:
        return sum(1 for _ in fh)


def group_elements(iterable, group_size, fillvalue=None):
    """
    Collect data into fixed-length chunks.
    """
    groups = [iter(iterable)] * group_size

    return zip_longest(*groups, fillvalue=fillvalue)


def save_examples(args, data_path, tokenizer):
    """
    Creates numericalizes examples from a raw WMT
    parallel corpora.
    """
    name = basename(splitext(data_path)[0])

    examples = read_file(data_path)

    groups = group_elements(
        iterable=examples,
        group_size=args.file_size)

    num_examples = 0
    filenames = []
    for idx, group in enumerate(groups):
        filename = join(
            args.data_dir,
            '{}{}.pt'.format(name, idx))

        filenames.append(filename)

        examples = list(generate_examples(group))
        torch.save({'dataset': examples}, filename)

    return filenames, num_examples


def save_lines(dump_path, line_generator):
    """
    Saves the generated lines into a file.
    """
    with open(dump_path, 'w') as f:
        for line in line_generator:
            f.write(line + '\n')


def generate_lines(file_handle, num_lines):
    """
    Generates `data_length` number of lines from a file.
    """
    for _, line in zip(range(num_lines), file_handle):
        yield line.strip()


def generate_examples(lines):
    """
    Generates id examples from dialogues.
    """
    for example in lines:
        if example is None:
            # reaching the last segment of the last
            # file so we can break from the loop
            break

        source, target = example

        yield source, target


def generate_splits(args, splits):
    """
    Creates from the downloaded WMT datafile.
    """
    data_path = join(args.data_dir, 'wmt16_en_de')

    data_size = compute_lines(data_path)

    with open(data_path, 'r') as fh:
        for filename, split_size in splits:
            num_lines = int(data_size * split_size)
            dump_path = join(args.data_dir, filename)

            line_generator = generate_lines(
                file_handle=fh,
                num_lines=num_lines)

            save_lines(
                dump_path=dump_path,
                line_generator=line_generator)

            yield dump_path, num_lines


def read_file(data_path):
    """
    Reads the contents of a raw dailydialog file.
    """
    with open(data_path, 'r') as fh:
        for line in fh:
            # the source - target pairs are separated
            # by \t in the raw WMT files
            yield line.strip().split('\t')


class Tokenizer:

    def __init__(self):
        pass

    def encode(self, text):
        pass

    def decode(self, ids):
        pass

    def convert_tokens_to_ids(self, tokens):
        pass

    def convert_ids_to_tokens(self, ids):
        pass


def create_loader(args, filenames, tokenizer,
                  distributed, shuffle=False):
    """
    Creates a generator that iterates through the
    dataset.
    """
    # distributed training is used if the local
    # rank is not the default -1
    sampler_cls = DistributedSampler if \
        distributed else IndexSampler

    bucket_sampler_cls = create_sampler_cls(
        sampler_cls=sampler_cls)

    def load_examples():
        """
        Generator that loads examples from files
        lazily.
        """
        file_dataset = FileDataset(filenames)
        file_loader = DataLoader(
            file_dataset,
            collate_fn=lambda x: x[0])

        for examples, indices in file_loader:
            sampler = bucket_sampler_cls(
                indices, shuffle=shuffle)

            translation_dataset = TranslationDataset(
                examples=examples)

            example_loader = DataLoader(
                translation_dataset,
                batch_size=args.batch_size,
                num_workers=4, sampler=sampler,
                pin_memory=True,
                collate_fn=padded_collate)

            yield from example_loader

    return load_examples


class FileDataset(Dataset):
    """
    Dataset that contains filenames for loading
    lazily.
    """

    def __init__(self, filenames):
        self.filenames = filenames

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        dataset = torch.load(filename)

        examples = dataset['examples']

        return examples

    def __len__(self):
        return len(self.filenames)


class TranslationDataset(Dataset):
    """
    Fetches utterances from a list of examples.
    """

    def __init__(self, examples):
        self.examples = examples

    def __getitem__(self, idx):
        source, target = self.examples[idx]

        # returning nested lists for convenient
        # parameter passing to collate_fn
        return [source, target]

    def __len__(self):
        return len(self.examples)


class IndexSampler(Sampler):
    """
    Dummy class for sampling indices in range
    `len(data_source)`.
    """

    def __init__(self, data_source):
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)


def create_sampler_cls(sampler_cls):
    """
    Creates a bucketized sampler class.
    """
    class BucketSampler(sampler_cls):
        """
        Bucketized sampler that yields exclusive groups
        of indices based on the sequence length.
        """

        def __init__(self, data_source, bucket_size=1000,
                     shuffle=True):
            super().__init__(data_source)
            self.bucket_size = bucket_size
            self.shuffle = shuffle

            self.sorted = sorted(
                list(super().__iter__()),
                key=lambda i: data_source[i][2])

        def __iter__(self):
            # divides the data into bucket size segments
            # and only these segment are shuffled
            segments = [
                self.sorted[idx: idx + self.bucket_size]
                for idx in range(0, len(self.sorted),
                                 self.bucket_size)]

            # selecting seqgemnts in random order
            random.shuffle(segments)
            for segment in segments:

                if self.shuffle:
                    random.shuffle(segment)

                yield from segment

    return BucketSampler


def create_dataset(args, device, distributed):
    """
    Downloads the DailyDialog dataset, converts it
    to tokens and returns iterators over the train and
    test splits.
    """
    metadata_path = join(
        args.data_dir, 'metadata.json')

    if args.force_new:
        shutil.rmtree(metadata_path)

    if not exists(metadata_path):
        # if dataset does not exist then create it
        # downloading and tokenizing the raw files
        download(args)

        tokenizer = Tokenizer()

        transformed = transform(args, tokenizer)
        train, valid, test = transformed

        train_files, train_size = train
        valid_files, valid_size = valid
        test_files, test_size = test

        print('Saving metadata to {}'.format(metadata_path))
        # save the location of the files in a metadata
        # json object and delete the file in case of
        # interrupt so it wont be left in corrupted state
        with open(metadata_path, 'w') as fh:
            try:
                json.dump({
                    'train': [train_files, train_size],
                    'valid': [valid_files, valid_size],
                    'test': [test_files, test_size]
                }, fh)
            except KeyboardInterrupt:
                shutil.rmtree(metadata_path)

    else:
        print('Loading metadata from {}'.format(
            metadata_path))
        with open(metadata_path, 'r') as fh:
            filenames = json.load(fh)

        train_files, train_size = filenames['train']
        valid_files, valid_size = filenames['valid']
        test_files, test_size = filenames['test']

        tokenizer = Tokenizer()

    train_dataset = create_loader(
        args=args,
        filenames=train_files,
        tokenizer=tokenizer,
        distributed=distributed,
        shuffle=True)

    valid_dataset = create_loader(
        args=args,
        filenames=valid_files,
        distributed=distributed,
        tokenizer=tokenizer)

    test_dataset = create_loader(
        args=args,
        filenames=test_files,
        distributed=distributed,
        tokenizer=tokenizer)

    train = train_dataset, train_size
    valid = valid_dataset, valid_size
    test = test_dataset, test_size

    return (train, valid, test), tokenizer
