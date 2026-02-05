import os
import csv
import torch
from torch.utils.data import Dataset
import torchaudio  

# -----------------------------------------------------------------------------
# DATASET IMPLEMENTATION NOTES
#
# - The dataset may return multiple features (e.g., waveform, spectrogram,
#   loudness estimate, embeddings, etc.).
#
# - All returned features must have consistent shapes within a batch.
#   Different feature types may have different shapes from each other, but the
#   *same* feature type must have the *same shape* across all samples in the batch.
#   (Example: if feature1 is a raw waveform, each feature 1 should be 
#   padded/cropped to N samples.
#
# - If applying random transforms (noise, time-shift, gain, etc.), only do so
#   when `self.training_flag` is True to ensure evaluation is deterministic.
#
# REQUIREMENTS FOR EVALUATION:
# We will test you dataset. Ensure the following:
#   1) The label is the last returned item. 
#      e.g. return feature1, feature2, label
#   2) All returned items are PyTorch tensors.
#   3) `self.class_names` contains the sorted unique class names.
#   4) Any augmentation or preprocessing happens inside the dataset, not in
#      external training/evaluation loops.
# -----------------------------------------------------------------------------


class LoadAudio(Dataset):
    def __init__(self, root_dir, meta_filename, audio_subdir, training_flag: bool = True):
        """
        Args:
            root_dir (str): Dataset root directory.
            meta_filename (str): Metadata filename inside root_dir.
            audio_subdir (str): Audio subdirectory relative to root_dir.
            training_flag (bool): When True, random transforms may be applied
                                  inside __getitem__ for data augmentation.
        """
        # 1) Store the directories/paths.
        # 2) Scan audio_subdir for candidate files.
        # 3) Read metadata: filename + label string → keep only valid files.
        # 4) Construct `self.class_names` (sorted unique labels) and then
        #    `self.label_to_idx` (class_name → integer index).
        # 5) Store samples as list of (filepath, label_string).

        meta_path = os.path.join(root_dir, meta_filename)
        audio_dir = os.path.join(root_dir, audio_subdir)

        # Reading the meta.txt
        samples = []
        class_set = set()
        with open(meta_path,"r") as f:
          reader =csv.reader(f,delimiter="\t")
          for row in reader:
            rel_path,label,clip_id = row
            full_path = os.path.join(root_dir, rel_path)
            if os.path.isfile(full_path):
              samples.append((full_path, label))
              class_set.add(label)

        self.samples = samples
        self.class_names = sorted(list(class_set))
        self.num_classes = len(self.class_names)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_names)}
        
        self.waveform_shape = (1, 220500)     # ~5 seconds at ~44.1kHz
        self.training_flag = training_flag

        # Mel Spectrogram transformer
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
          sample_rate=44100,
          n_fft=1024,
          hop_length=512,
          n_mels=80
        )
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=44100,
            n_mfcc=80,
            melkwargs={
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80,
            },
        )
         # Convert to decibels (log-mel)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Steps to implement:
        #   1) filepath, label_str = self.samples[idx]
        #   2) waveform, sr = torchaudio.load(filepath)
        #   3) If self.training_flag is True, apply augmentations here
        #   4) Ensure waveform shape is consistent (crop/pad if necessary)
        #   5) label_idx = self.label_to_idx[label_str]
        #   6) Return (feature(s), label_idx) with the label as the final item.

        # Placeholder output for now
        filepath, label_str = self.samples[idx]
        # Load audio
        waveform, sr = torchaudio.load(filepath)   # shape: [C, T]

        # Convert stereo → mono if needed
        if waveform.shape[0] > 1:
          waveform = waveform.mean(dim=0, keepdim=True)
        # Pad or crop to fixed length
        target_len = self.waveform_shape[1]
        if waveform.shape[1] < target_len:
          pad_len = target_len - waveform.shape[1]
          waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        else:
          waveform = waveform[:, :target_len]
        # Convert waveform → mel spectrogram
        mel = self.mel_transform(waveform)    
        mel = self.amplitude_to_db(mel)
        # Convert waveform → MFCC (same frequency resolution, different representation)
        mfcc = self.mfcc_transform(waveform)      
         # Remove the extra channel dim so we can stack channels ourselves
        mel = mel.squeeze(0)    
        mfcc = mfcc.squeeze(0)  
         # Stack into a 2-channel "image": channel 0 = mel, channel 1 = MFCC
        features = torch.stack([mel, mfcc],dim=0)  
        # Optional light augmentation (training only)
        if self.training_flag:
          features = features + 0.007 * torch.randn_like(features)
          # SpecAugment: mask random time segments and frequency bands
        label_idx = self.label_to_idx[label_str]
        return features, label_idx
