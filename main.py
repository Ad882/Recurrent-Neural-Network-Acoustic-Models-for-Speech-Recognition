from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torchaudio
import torch
import pyfiglet

model_name = "openai/whisper-base"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)
model.load_state_dict(torch.load("model/whisper_model.pth", weights_only=True))
model.eval()

ascii_banner = pyfiglet.figlet_format("ASR")
print(ascii_banner)

def preprocess_audio(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=16000)

    return waveform.squeeze(0)

audio_path = "data/audio_sample/common_voice_en_39177958.mp3"
waveform = preprocess_audio(audio_path)

inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model.generate(**inputs, language="en")

transcription = processor.batch_decode(logits, skip_special_tokens=True)[0]

print("Transcription :", transcription)