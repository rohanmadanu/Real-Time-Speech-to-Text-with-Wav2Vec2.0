# Real-Time-Speech-to-Text-with-Wav2Vec2.0
Lightweight, accurate speech-to-text using Wav2Vec2.0 via PyTorch/torchaudio. Runs on Colab CPU—handles .wav uploads, sample rate conversion &amp; instant transcription with Greedy CTC Decoder. No training needed. Great for ASR pipelines &amp; voice apps.
!pip install torchaudio --quiet

import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
import torch
import IPython.display as ipd
from google.colab import files

#  Load pretrained model
bundle = WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
labels = bundle.get_labels()

# Create a simple greedy decoder
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=labels)

#  Upload audio
print("Upload a .wav file with clear speech (16kHz recommended)")
uploaded = files.upload()
filepath = list(uploaded.keys())[0]

#  Load & play audio
waveform, sample_rate = torchaudio.load(filepath)
print("\n🎧 Playing your audio...")
ipd.display(ipd.Audio(filepath))

#  Resample if needed
if sample_rate != bundle.sample_rate:
    print(f" Resampling from {sample_rate}Hz to {bundle.sample_rate}Hz...")
    resampler = torchaudio.transforms.Resample(sample_rate, bundle.sample_rate)
    waveform = resampler(waveform)

#  Recognize speech
print("\n🔍 Processing speech...")
with torch.inference_mode():
    emissions, _ = model(waveform)
    
#  Decode and print transcription
transcription = decoder(emissions[0])
print("\n Transcription:", transcription.lower())
