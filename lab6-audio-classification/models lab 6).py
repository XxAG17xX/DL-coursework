import torch
import torch.nn as nn
import torchaudio

class AudioClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        #mel Spectrogram transforms (used only if input is raw waveform)
        # This is a fallback for when the tester feeds raw waveforms [B,1,T].
        # For normal training,we already compute mel-spectrograms in the dataset.
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100,
            n_fft=1024,
            hop_length=512,
            n_mels=80           # was 128, reduced for slightly less noisy features
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=44100,
            n_mfcc=80,
            melkwargs={
                "n_fft": 1024,
                "hop_length": 512,
                "n_mels": 80,
            },
        )
        # CNN Feature Extractor for mel spectrograms
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),      # mel: 64 → 32

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),      # 32 → 16

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),      # 16 → 8
        )

        #Adaptive pooling removes variable time dimension
        self.global_pool = nn.AdaptiveAvgPool2d((4,4))  # output → [128,4,4]

        #Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),              # → 128 * 4 * 4 = 2048
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        """
        x can be:
        - waveform: [B, 1, T]   → convert to mel
        - mel spec: [B, 1, 64, time] → use directly
        """
        # If input is [B,1,T] = raw waveform(tester/ummary case)
        if x.ndim == 3:
            # x: [B, 1, T]
            mel = self.mel_transform(x)          #[B, 1, 64, T']
            mel = self.amplitude_to_db(mel)
            mfcc = self.mfcc_transform(x)        #[B, 1, 64, T']

            #Remove the extra channel dim (C=1) and stack into channels=2
            mel = mel.squeeze(1)                 # [B, 64, T']
            mfcc = mfcc.squeeze(1)               # [B, 64, T']
            x = torch.stack([mel, mfcc], dim=1)  # [B, 2, 64, T']

        #If x.ndim == 4,we assume it is already [B, 2, 64, T]
        x = self.features(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x