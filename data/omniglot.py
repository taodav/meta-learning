import os
import numpy as np

import tensorflow_datasets as tfds

def omniglot_tf_dataset():
    train_data = tfds.load(name='omniglot', split="train")
    test_data = tfds.load(name='omniglot', split='test')
    return train_data, test_data

class OmniglotDataset:
    """
    Omniglot dataset, separated by alphabet.
    Dataset parsing referenced and inspired from
    https://github.com/Goldesel23/Siamese-Networks-for-One-Shot-Learning/blob/master/omniglot_loader.py
    """
    def __init__(self, data_path, augment_images=False, batch_size=32):
        """
        :param data_path: path to the omniglot dataset folder (https://github.com/brendenlake/omniglot)
        :param augment_images: should we include augmented images
        :param batch_size: size of a batch for minibatch sampling
        """
        self._data_path = data_path
        self._batch_size = batch_size
        self._augment = augment_images
        self._train_alphabets = []
        self._val_alphabets = []
        self._eval_alphabets = []
        self._curr_train_alphabet_index = 0
        self._curr_valid_alphabet_index = 0
        self._curr_test_alphabet_index = 0

        self.train_dictionary = dict()
        self.eval_dictionary = dict()

        self.load()

    @staticmethod
    def parse_alphabets(alphabets_path):
        data_dictionary = {}
        for alphabet in os.listdir(alphabets_path):
            alph_path = os.path.join(alphabets_path, alphabet)
            alph_dictionary = {}

            for char in os.listdir(alph_path):
                char_path = os.path.join(alph_path, char)
                alph_dictionary[char] = os.listdir(char_path)

            data_dictionary[alphabet] = alph_dictionary

        return data_dictionary


    def load(self):

        train_path = os.path.join(self._data_path, 'images_background')
        eval_path = os.path.join(self._data_path, 'images_evaluation')

        self.train_dictionary = self.parse_alphabets(train_path)
        self.eval_dictionary = self.parse_alphabets(eval_path)

    def train_batch(self, imgs_per_datum=1):
        """
        Get a batch from the current alphabet
        :return:
        """
        curr_alph = self._train_alphabets[self._curr_train_alphabet_index]
        avail_chars = list(self.train_dictionary[curr_alph])

        batch_paths = []
        batch_char_indices = np.random.randint(0, high=len(avail_chars), size=self._batch_size)

        uniform_sample_prob = 1 / (len(avail_chars) - 1)
        char_probs = np.ones_like(avail_chars) * uniform_sample_prob
        char_indices = np.arange(0, len(avail_chars))

        for index in batch_char_indices:
            curr_char = avail_chars[index]
            avail_images = self.train_dictionary[curr_alph][curr_char]
            char_dir_path = os.path.join(self._data_path, 'images_background', curr_alph, curr_char)

            # sample a random pair of character images
            datum = []
            image_indices = np.random.choice(len(avail_images), imgs_per_datum)
            for img_idx in image_indices:
                image_path = os.path.join(char_dir_path, avail_images[img_idx])
                datum.append(img_idx)

            # Instead of copying an array and deleting we just sample with equal prob
            # and 0 prob in the index that we want
            char_probs[index] = 0

            diff_char = avail_chars[np.random.choice(char_indices, p=char_probs)]
            diff_img_path = os.path.join(char_dir_path, diff_char)
            datum.append(diff_img_path)

            batch_paths.append(datum)

        self._curr_train_alphabet_index += 1

        if self._curr_train_alphabet_index > len(self._train_alphabets):
            self._curr_train_alphabet_index = 0

        images, labels = self._convert_batch_paths_to_images(batch_paths)

        return images, labels

    def _convert_batch_paths_to_images(self, batch_paths):
        # TODO
        pass


if __name__ == "__main__":
    from definitions import ROOT_DIR
    dset_path = os.path.join(ROOT_DIR, 'data', 'datasets')
    dset = OmniglotDataset(dset_path)

    dset.train_batch()
