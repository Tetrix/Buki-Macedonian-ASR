The models are developed using the SpeechBrain toolkit.

To train the models, you need to run: `python train.py hyperparams.yaml`

The trained models are available on Huggingface:
Wav2vec2: https://huggingface.co/Macedonian-ASR/buki-wav2vec2-2.0/tree/main
Whisper: https://huggingface.co/Macedonian-ASR/buki-whisper-2.0

To run inference with the Whisper model, use:

```
from speechbrain.inference.interfaces import foreign_class
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
asr_classifier = foreign_class(source="Macedonian-ASR/buki-whisper-2.0", pymodule_file="custom_interface.py", classname="ASR")
asr_classifier = asr_classifier.to(device)
predictions = asr_classifier.classify_file("audio_file.wav", device)
print(predictions)
```




