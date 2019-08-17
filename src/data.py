"""
@author:    Patrik Purgai
@copyright: Copyright 2019, supervised-translation
@license:   MIT
@email:     purgai.patrik@gmail.com
@date:      2019.04.04.
"""

# pylint: disable=import-error

import torch
import requests
import shutil
import os
import random
import json

import sentencepiece as spm

from collate import padded_collate

from tqdm import tqdm
from itertools import (
    zip_longest, chain)

from joblib import Parallel, delayed

from os.path import (
    exists, join,
    dirname, abspath,
    basename, splitext)

from torch.utils.data import (
    Dataset, DataLoader,
    Sampler)

from torch.utils.data.distributed import (
    DistributedSampler)


SOS = '<sos>'
EOS = '<eos>'
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
        default=32000,
        help='Maximum size of the vocabulary.')
    parser.add_argument(
        '--file_size',
        type=int,
        default=100000,
        help='Max number of examples in a single file.')
    parser.add_argument(
        '--max_len',
        type=int,
        default=50,
        help='Maximum length of a sequence.')
    parser.add_argument(
        '--lower',
        type=bool,
        default=True,
        help='Sentences are lowered.')
    parser.add_argument(
        '--max_sentences',
        type=int,
        default=1000000,
        help='Maximum number of sentences for tokenizer.')


def download(args):
    """
    Downloads and extracts the daily dialog dataset from 
    google drive.
    """
    base_url = 'http://www.statmt.org/europarl/v7/'
    filename = 'de-en.tgz'

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.download_dir, exist_ok=True)

    url = base_url + filename
    download_path = join(args.download_dir, filename)

    if not exists(download_path):
        print('Downloading dataset to {}'.format(
            download_path))

        with requests.Session() as session:
            response = session.get(
                url, stream=True, timeout=5)

            # data is read in 2 ** 15 sized chunks
            # NOTE this could be tuned to reveal
            # data size in MBs
            loop = response.iter_content(2 ** 15)

            with open(download_path, 'wb') as f:
                for chunk in tqdm(loop):
                    if chunk:
                        f.write(chunk)

    source_file = join(args.data_dir,
                       'europarl-v7.de-en.en')
    target_file = join(args.data_dir,
                       'europarl-v7.de-en.de')

    if not exists(source_file) or not exists(target_file):
        print('Extracting dataset to {}'.format(
            args.data_dir))
        shutil.unpack_archive(
            download_path, args.data_dir)
        
        if args.lower:
            source = list(read_file(args, source_file))
            with open(source_file, 'w') as fh:
                fh.write('\n'.join(source))

            target = list(read_file(args, target_file))
            with open(target_file, 'w') as fh:
                fh.write('\n'.join(target))

    return source_file, target_file


def create_tokenizer(args, prefix, data_path):
    """
    Creates tokenizers from raw datafiles.
    """
    tokenizer_path = join(args.data_dir, prefix)

    if not exists(tokenizer_path + '.model'):
        param_string = ' '.join([
            '--input={data_path}',
            '--vocab_size={vocab_size}',
            '--model_prefix={prefix}',
            '--pad_id=0',
            '--pad_piece={pad}',
            '--unk_id=1',
            '--unk_piece={unk}',
            '--bos_id=2',
            '--bos_piece={sos}',
            '--eos_id=3',
            '--eos_piece={eos}',
            '--num_threads=4',
            '--input_sentence_size={max_sent}'
            '--shuffle_input_sentence=True'
        ]).format(
            data_path=data_path,
            vocab_size=args.vocab_size,
            prefix=tokenizer_path,
            max_sent=args.max_sentences,
            pad=PAD, unk=UNK,
            sos=SOS, eos=EOS)

        spm.SentencePieceTrainer.train(param_string)

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path + '.model')

    return tokenizer


def transform(args, data_files):
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
        (save_examples(args=args,
                       data_path=filename), size)
        for filename, size in
        generate_splits(args, data_files, splits)]

    return train, valid, test


def compute_lines(filename):
    """
    Computes the number of lines of a file.
    """
    with open(filename, 'r') as fh:
        return sum(1 for _ in fh)


def group_elements(iterable, group_size):
    """
    Collect data into fixed-length chunks.
    """
    groups = [iter(iterable)] * group_size

    return zip_longest(*groups)


def save_examples(args, data_path):
    """
    Creates numericalizes examples from a raw WMT
    parallel corpora.
    """
    name = basename(splitext(data_path)[0])

    examples = read_file(args, data_path)

    groups = group_elements(
        iterable=examples,
        group_size=args.file_size)

    filenames = []
    for idx, group in enumerate(groups):
        filename = join(
            args.data_dir,
            '{}{}.pt'.format(name, idx))

        filenames.append(filename)

        examples = list(generate_examples(group))
        torch.save({'examples': examples}, filename)

    return filenames


def save_lines(dump_path, line_generator):
    """
    Saves the generated lines into a file.
    """
    with open(dump_path, 'w') as f:
        for line in line_generator:
            f.write('\t'.join(line) + '\n')


def generate_lines(file_handle, num_lines):
    """
    Generates `data_length` number of lines from a file.
    """
    for _, line in zip(range(num_lines), file_handle):
        yield line


def generate_examples(lines):
    """
    Generates examples from groups.
    """
    for example in lines:
        if example is None:
            # reaching the last segment of the last
            # file so we can break from the loop
            break

        try:
            source, target = example.split('\t')
            yield source, target

        except ValueError:
            pass


def generate_splits(args, data_files, splits):
    """
    Creates splits from the downloaded WMT datafile.
    """
    source_file, target_file = data_files
    data_size = compute_lines(source_file)

    data_files = zip(
        read_file(args, source_file),
        read_file(args, target_file))

    for filename, split_size in splits:
        num_lines = int(data_size * split_size)
        dump_path = join(args.data_dir, filename)

        line_generator = tqdm(
            generate_lines(
                file_handle=data_files,
                num_lines=num_lines))

        save_lines(
            dump_path=dump_path,
            line_generator=line_generator)

        yield dump_path, num_lines


def read_file(args, data_path):
    """
    Reads the contents of a raw europarl files.
    """
    with open(data_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line != '':
                if args.lower:
                    line = line.lower()

                yield line


def create_loader(args, filenames, tokenizers,
                  shuffle=False, sampled=False):
    """
    Creates a generator that iterates through the
    dataset.
    """
    # distributed training is used if the local
    # rank is not the default -1
    sampler_cls = DistributedSampler if \
        args.local_rank == -1 else IndexSampler

    bucket_sampler_cls = create_sampler_cls(
        sampler_cls=sampler_cls)

    def load_examples():
        """
        Generator that loads examples from files
        lazily.
        """
        file_dataset = FileDataset(
            filenames,
            tokenizers=tokenizers,
            sampled=sampled)

        file_loader = DataLoader(
            file_dataset,
            collate_fn=lambda x: x[0])

        for examples in file_loader:
            sampler = bucket_sampler_cls(
                examples, shuffle=shuffle)

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

    def __init__(self, filenames, tokenizers,
                 sampled):
        self.filenames = filenames
        self.tokenizers = tokenizers
        self.sampled = sampled

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        dataset = torch.load(filename)

        raw_examples = dataset['examples']
        tokenized_examples = []

        source_tokenizer, target_tokenizer = \
            self.tokenizers

        for source, target in raw_examples:
            # sample from tokenizer encoding during
            # training to apply subword reguralization

            if self.sampled:
                numericalized = (
                    source_tokenizer.sample_encode_as_ids(
                        input=source, nbest_size=-1, 
                        alpha=0.1),
                    target_tokenizer.sample_encode_as_ids(
                        input=target, nbest_size=-1, 
                        alpha=0.1))
            else:
                numericalized = (
                    source_tokenizer.encode_as_ids(source),
                    target_tokenizer.encode_as_ids(target))

            tokenized_examples.append(numericalized)

        return tokenized_examples

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
                key=lambda i: (
                    len(data_source[i][0]),
                    len(data_source[i][1])))

        def __iter__(self):
            # divides the data into bucket size segments
            # and only these segment are shuffled
            segments = [
                self.sorted[idx: idx + self.bucket_size]
                for idx in range(0, len(self.sorted),
                                 self.bucket_size)]

            # selecting segements in random order
            random.shuffle(segments)
            for segment in segments:

                if self.shuffle:
                    random.shuffle(segment)

                yield from segment

    return BucketSampler


def create_dataset(args, device):
    """
    Downloads the DailyDialog dataset, converts it
    to tokens and returns iterators over the train and
    test splits.
    """
    metadata_path = join(
        args.data_dir, 'metadata.json')

    # data is only downloaded if it is not found in
    # `args.data_dir` directory
    source_file, target_file = download(args)

    source_tokenizer = create_tokenizer(
        args=args, prefix='source',
        data_path=source_file)

    target_tokenizer = create_tokenizer(
        args=args, prefix='target',
        data_path=target_file)

    tokenizers = source_tokenizer, target_tokenizer

    if not exists(metadata_path):
        # if dataset does not exist then create it
        # downloading and tokenizing the raw files
        data_files = source_file, target_file
        train, valid, test = transform(args, data_files)

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

    # shuffle dataset and use subword sampling
    # reguralization only during training
    train_dataset = create_loader(
        args=args,
        filenames=train_files,
        tokenizers=tokenizers,
        sampled=True,
        shuffle=True)

    valid_dataset = create_loader(
        args=args,
        filenames=test_files,
        tokenizers=tokenizers)

    test_dataset = create_loader(
        args=args,
        filenames=test_files,
        tokenizers=tokenizers)

    train = train_dataset, train_size
    valid = valid_dataset, valid_size
    test = test_dataset, test_size

    return (train, valid, test), tokenizers
