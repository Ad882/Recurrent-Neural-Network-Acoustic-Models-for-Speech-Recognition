from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
import torch

model_name = "openai/whisper-base"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)


torch.save(model.state_dict(), "model/whisper_model.pth")
