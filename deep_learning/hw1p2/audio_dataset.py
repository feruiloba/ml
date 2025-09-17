import torch
import torch.nn as nn
import numpy as np
import gc
import zipfile
import bisect
from tqdm.auto import tqdm
import os
import datetime
import wandb
import yaml
import torchaudio.transforms as tat
import torchaudio

PHONEMES = [
            '[SIL]',   'AA',    'AE',    'AH',    'AO',    'AW',    'AY',
            'B',     'CH',    'D',     'DH',    'EH',    'ER',    'EY',
            'F',     'G',     'HH',    'IH',    'IY',    'JH',    'K',
            'L',     'M',     'N',     'NG',    'OW',    'OY',    'P',
            'R',     'S',     'SH',    'T',     'TH',    'UH',    'UW',
            'V',     'W',     'Y',     'Z',     'ZH',    '[SOS]', '[EOS]']

# Dataset class to load train and validation data

config = {
    'Name': 'Fernando\'s model config', # Write your name here
    'subset': 0.0009, # Subset of train/val dataset to use (1.0 == 100% of data)
    'context': 30,
    'archetype': 'diamond', # Default Values: pyramid, diamond, inverse-pyramid,cylinder
    'activations': 'GELU',
    'learning_rate': 0.001,
    'dropout': 0.25,
    'optimizers': 'SGD',
    'scheduler': 'ReduceLROnPlateau',
    'epochs': 1,
    'batch_size': 32,
    'weight_decay': 0.05,
    'weight_initialization': None, # e.g kaiming_normal, kaiming_uniform, uniform, xavier_normal or xavier_uniform
    'augmentations': 'Both', # Options: ["FreqMask", "TimeMask", "Both", null]
    'freq_mask_param': 4,
    'time_mask_param': 8
 }

class AudioDataset(torch.utils.data.Dataset):

    def __init__(self, root, phonemes = PHONEMES, context=0, partition= "train-clean-100"): # Feel free to add more arguments

        self.context    = context
        self.phonemes   = phonemes
        self.subset = config['subset']

        # Initialize augmentations. Read the Pytorch torchaudio documentations on timemasking and frequencymasking
        self.freq_masking = tat.FrequencyMasking(28)
        self.time_masking = tat.TimeMasking(time_mask_param = 25)

        # MFCC directory - use partition to acces train/dev directories from kaggle data using root
        self.mfcc_dir       = f'{root}/{partition}/mfcc'
        # Transcripts directory - use partition to acces train/dev directories from kaggle data using root
        self.transcript_dir = f'{root}/{partition}/transcript'

        # List files in sefl.mfcc_dir using os.listdir in SORTED order
        mfcc_names          = os.listdir(self.mfcc_dir)
        # List files in self.transcript_dir using os.listdir in SORTED order
        transcript_names    = os.listdir(self.transcript_dir)

        # Compute size of data subset
        subset_size = int(self.subset * len(mfcc_names))

        # Select subset of data to use
        mfcc_names = mfcc_names[:subset_size]
        transcript_names = transcript_names[:subset_size]

        # Making sure that we have the same no. of mfcc and transcripts
        assert len(mfcc_names) == len(transcript_names)

        self.mfccs, self.transcripts = [], []


        # Iterate through mfccs and transcripts
        for i in tqdm(range(len(mfcc_names))):

            # Load a single mfcc. Hint: Use numpy
            mfcc             = np.load(f'{root}/{partition}/mfcc/{mfcc_names[i]}')
            # Do Cepstral Normalization of mfcc along the Time Dimension (Think about the correct axis)
            mfccs_normalized = mfcc - np.mean(mfcc, axis=0) / (np.std(mfcc, axis=0) + 1e-9)

            # Convert mfcc to tensor
            mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

            # Load the corresponding transcript
            # Remove [SOS] and [EOS] from the transcript
            # (Is there an efficient way to do this without traversing through the transcript?)
            # Note that SOS will always be in the starting and EOS at end, as the name suggests.

            transcript  = np.load(f'{root}/{partition}/transcript/{transcript_names[i]}')[1:-1]

            # The available phonemes in the transcript are of string data type
            # But the neural network cannot predict strings as such.
            # Hence, we map these phonemes to integers

            # Map the phonemes to their corresponding list indexes in self.phonemes
            transcript_indices = [self.phonemes.index(t) for t in transcript if t in self.phonemes]
            # Now, if an element in the transcript is 0, it means that it is 'SIL' (as per the above example)

            # Convert transcript to tensor
            transcript_indices = torch.tensor(transcript_indices, dtype=torch.int64)

            # Append each mfcc to self.mfcc, transcript to self.transcript
            self.mfccs.append(mfccs_normalized)
            self.transcripts.append(transcript_indices)

        # NOTE:
        # Each mfcc is of shape T1 x 28, T2 x 28, ...
        # Each transcript is of shape (T1+2), (T2+2) before removing [SOS] and [EOS]

        # TODO: Concatenate all mfccs in self.mfccs such that
        # the final shape is T x 28 (Where T = T1 + T2 + ...)
        # Hint: Use torch to concatenate
        self.mfccs          = torch.cat(self.mfccs, axis=0)

        # TODO: Concatenate all transcripts in self.transcripts such that
        # the final shape is (T,) meaning, each time step has one phoneme output
        # Hint: Use torch to concatenate
        self.transcripts    = torch.cat(self.transcripts, axis=0)

        # Length of the dataset is now the length of concatenated mfccs/transcripts
        self.length = len(self.mfccs)

        # Take some time to think about what we have done.
        # self.mfcc is an array of the format (Frames x Features).
        # Our goal is to recognize phonemes of each frame

        # We can introduce context by padding zeros on top and bottom of self.mfcc
        # Hint: Use torch.nn.functional.pad
        # torch.nn.functional.pad takes the padding in the form of (left, right, top, bottom) for 2D data
        self.mfccs = torch.nn.functional.pad(self.mfccs, (0, 0, context, context))


    def __len__(self):
        return self.length

    def collate_fn(self, batch):
      x, y = zip(*batch)
      x = torch.stack(x, dim=0)

      # Apply augmentations with 70% probability (You can modify the probability)
      if np.random.rand() < 0.70:
        x = x.transpose(1, 2)  # Shape: (batch_size, freq, time)
        x = self.freq_masking(x)
        x = self.time_masking(x)
        x = x.transpose(1, 2)  # Shape back to: (batch_size, time, freq)

      return x, torch.tensor(y)

    def __getitem__(self, ind):
        # Based on context and offset, return a frame at given index with context frames to the left, and right.
        frames = self.mfccs[ind:ind+2*self.context+1]

        # After slicing, you get an array of shape 2*context+1 x 28.

        phonemes = self.transcripts[ind]

        return frames, phonemes


# Dataset class to load test data
# Create a test dataset class similar to the previous class but you dont have transcripts for this
# Read the mfccs in sorted order, do NOT shuffle the data here or in your dataloader.
# IMPORTANT: Load complete test data to use, DO NOT select subset of test data, else you will get errors when submitting on Kaggle.

class AudioTestDataset(torch.utils.data.Dataset):

    def __init__(self, root, context=0, partition= "test-clean"):

      self.context    = context

      # Initialize augmentations. Read the Pytorch torchaudio documentations on timemasking and frequencymasking
      self.freq_masking = tat.FrequencyMasking(28)
      self.time_masking = tat.TimeMasking(time_mask_param = 25)

      # MFCC directory - use partition to acces train/dev directories from kaggle data using root
      self.mfcc_dir       = f'{root}/{partition}/mfcc'

      # List files in sefl.mfcc_dir using os.listdir in SORTED order
      mfcc_names          = os.listdir(self.mfcc_dir)
      self.mfccs = []


      # Iterate through mfccs and transcripts
      for i in tqdm(range(len(mfcc_names))):

          # Load a single mfcc. Hint: Use numpy
          mfcc             = np.load(f'{root}/{partition}/mfcc/{mfcc_names[i]}')
          # Do Cepstral Normalization of mfcc along the Time Dimension (Think about the correct axis)
          mfccs_normalized = mfcc - np.mean(mfcc, axis=0) / (np.std(mfcc, axis=0) + 1e-9)

          # Convert mfcc to tensor
          mfccs_normalized = torch.tensor(mfccs_normalized, dtype=torch.float32)

          # Append each mfcc to self.mfcc, transcript to self.transcript
          self.mfccs.append(mfccs_normalized)

      # NOTE:
      # Each mfcc is of shape T1 x 28, T2 x 28, ...
      # Each transcript is of shape (T1+2), (T2+2) before removing [SOS] and [EOS]

      # TODO: Concatenate all mfccs in self.mfccs such that
      # the final shape is T x 28 (Where T = T1 + T2 + ...)
      # Hint: Use torch to concatenate
      self.mfccs          = torch.cat(self.mfccs, axis=0)

      # Length of the dataset is now the length of concatenated mfccs/transcripts
      self.length = len(self.mfccs)

      # Take some time to think about what we have done.
      # self.mfcc is an array of the format (Frames x Features).
      # Our goal is to recognize phonemes of each frame

      # We can introduce context by padding zeros on top and bottom of self.mfcc
      # Hint: Use torch.nn.functional.pad
      # torch.nn.functional.pad takes the padding in the form of (left, right, top, bottom) for 2D data
      self.mfccs = torch.nn.functional.pad(self.mfccs, (0, 0, 0, len(self.mfccs) % context))

    def __len__(self):
      return self.length

    def __getitem__(self, ind):
        # Based on context and offset, return a frame at given index with context frames to the left, and right.
        frames = self.mfccs[ind:ind+2*self.context+1]

        return frames

