import spaces
import os
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import gc
from functools import partial
import gradio as gr
import torch
from speechbrain.inference.interfaces import Pretrained, foreign_class
from transformers import T5Tokenizer, T5ForConditionalGeneration
import librosa
import whisper_timestamped as whisper
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline, Wav2Vec2ForCTC, AutoProcessor
from speechbrain.inference.VAD import VAD
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True

# Load the VAD model
vad_model = VAD.from_hparams(
    source="speechbrain/vad-crdnn-libriparty",
    savedir="vad_model",
    )

def clean_up_memory():
    gc.collect()
    torch.cuda.empty_cache()


@spaces.GPU(duration=15)
def recap_sentence(string):
    # Restore capitalization and punctuation using the model
    inputs = recap_tokenizer(["restore capitalization and punctuation: " + string], return_tensors="pt", padding=True).to(device)
    outputs = recap_model.generate(**inputs, max_length=768, num_beams=5, early_stopping=True).squeeze(0)
    recap_result = recap_tokenizer.decode(outputs, skip_special_tokens=True)
    return recap_result


@spaces.GPU(duration=30)
def return_prediction_whisper_mic(mic=None, vad_model=vad_model, device=device):
    if mic is not None:
        download_path = mic.split(".")[0] + ".txt"
        whisper_result = whisper_classifier.classify_file_whisper_mkd(mic, vad_model, device)
    else:
        whisper_result = ""
        download_path = "empty.txt"
        with open(download_path, "w") as f:
            f.write(whisper_result)
        yield whisper_result, download_path
    
    recap_result = ""
    prev_segment = ""
    prev_segment_len = 0

    for k, segment in enumerate(whisper_result):
        if prev_segment == "":
            recap_segment= recap_sentence(segment[0])
        else:
            prev_segment_len = len(prev_segment.split())
            recap_segment = recap_sentence(prev_segment + " " + segment[0])
        # remove prev_segment from the beginning of the recap_result
        recap_segment = recap_segment.split()
        recap_segment = recap_segment[prev_segment_len:]
        recap_segment = " ".join(recap_segment)
        prev_segment = segment[0]
        recap_result += recap_segment + " "

        # If the letter after punct is small, recap it
        for i, letter in enumerate(recap_result):
            if i > 1 and recap_result[i-2] in [".", "!", "?"] and letter.islower():
                recap_result = recap_result[:i] + letter.upper() + recap_result[i+1:]

        clean_up_memory()

        with open(download_path, "w") as f:
            f.write(recap_result)
    
        yield recap_result, download_path


@spaces.GPU(duration=30)
def return_prediction_whisper_file(file=None, vad_model=vad_model, device=device):
    whisper_result = []
    if file is not None:
        download_path = file.split(".")[0] + ".txt"
        whisper_result = whisper_classifier.classify_file_whisper_mkd(file, vad_model, device)
    else:
        whisper_result = ""
        download_path = "empty.txt"
        with open(download_path, "w") as f:
            f.write(whisper_result)
        yield whisper_result, download_path
    
    recap_result = ""
    prev_segment = ""
    prev_segment_len = 0

    for k, segment in enumerate(whisper_result):
        if prev_segment == "":
            recap_segment= recap_sentence(segment[0])
        else:
            prev_segment_len = len(prev_segment.split())
            recap_segment = recap_sentence(prev_segment + " " + segment[0])
        # remove prev_segment from the beginning of the recap_result
        recap_segment = recap_segment.split()
        recap_segment = recap_segment[prev_segment_len:]
        recap_segment = " ".join(recap_segment)
        prev_segment = segment[0]
        recap_result += recap_segment + " "

        # If the letter after punct is small, recap it
        for i, letter in enumerate(recap_result):
            if i > 1 and recap_result[i-2] in [".", "!", "?"] and letter.islower():
                recap_result = recap_result[:i] + letter.upper() + recap_result[i+1:]

        clean_up_memory()

        with open(download_path, "w") as f:
            f.write(recap_result)
    
        yield recap_result, download_path


# Create a partial function with the device pre-applied
return_prediction_whisper_mic_with_device = partial(return_prediction_whisper_mic, vad_model=vad_model, device=device)
return_prediction_whisper_file_with_device = partial(return_prediction_whisper_file, vad_model=vad_model, device=device)

# Load the ASR models
whisper_classifier = foreign_class(source="Macedonian-ASR/buki-whisper-2.0", pymodule_file="custom_interface_app.py", classname="ASR")
whisper_classifier = whisper_classifier.to(device)
whisper_classifier.eval()

# Load the T5 tokenizer and model for restoring capitalization
recap_model_name = "Macedonian-ASR/mt5-restore-capitalization-macedonian"
recap_tokenizer = T5Tokenizer.from_pretrained(recap_model_name)
recap_model = T5ForConditionalGeneration.from_pretrained(recap_model_name, torch_dtype=torch.float16)
recap_model.to(device)
recap_model.eval()


mic_transcribe_whisper = gr.Interface(
    fn=return_prediction_whisper_mic_with_device,
    inputs=gr.Audio(sources="microphone", type="filepath"),
    outputs=[gr.Textbox(label="Транскрипција"), gr.File(label="Зачувај го транскриптот", file_count="single")],
    allow_flagging="never",
    live=True
)

file_transcribe_whisper = gr.Interface(
    fn=return_prediction_whisper_file_with_device,
    inputs=gr.Audio(sources="upload", type="filepath"),
    outputs=[gr.Textbox(label="Транскрипција"), gr.File(label="Зачувај го транскриптот", file_count="single")],
    allow_flagging="never",
    live=True
)

project_description_header = '''
<div class="header">
    <img src="https://i.ibb.co/hYhkkhg/Buki-logo-1.jpg"
         alt="Bookie logo"
         style="float: right; width: 150px; height: 150px; margin-left: 10px;" />
    <img src="https://i.ibb.co/HpHg7281/qr-buki-whisper.png"
         alt="Bookie QR"
         style="float: right; width: 150px; height: 150px; margin-left: 10px;" />
    
    <h2>Автори:</h2>
    <ol>
        <li>Дејан Порјазовски</li>
        <li>Илина Јакимовска</li>
        <li>Ордан Чукалиев</li>
        <li>Никола Стиков</li>
   
    <h4>Оваа колаборација е дел од активностите на Фондација <a href="https://qantarot.substack.com/about"><strong>КАНТАРОТ</strong></a> и <strong>Центарот за напредни интердисциплинарни истражувања (<a href="https://ukim.edu.mk/en/centri/centar-za-napredni-interdisciplinarni-istrazhuvanja-ceniis">ЦеНИИс</a>)</strong> при УКИМ.</h4>
</div>
'''

project_description_footer = '''
<div class="footer">
    <h2>Во тренирањето на овој модел се употребени податоци од:</h2>
    <ol>
        <li>Дигитален архив за етнолошки и антрополошки ресурси (<a href="https://iea.pmf.ukim.edu.mk/tabs/view/61f236ed7d95176b747c20566ddbda1a">ДАЕАР</a>) при Институтот за етнологија и антропологија, Природно-математички факултет при УКИМ.</li>
        <li>Аудио верзија на меѓународното списание <a href="https://etno.pmf.ukim.mk/index.php/eaz/issue/archive">„ЕтноАнтропоЗум"</a> на Институтот за етнологија и антропологија, Природно-математички факултет при УКИМ.</li>
        <li>Аудио подкастот <a href="https://obicniluge.mk/episodes/">„Обични луѓе"</a> на Илина Јакимовска</li>
        <li>Научните видеа од серијалот <a href="http://naukazadeca.mk">„Наука за деца"</a>, фондација <a href="https://qantarot.substack.com/">КАНТАРОТ</a></li>
        <li>Македонска верзија на <a href="https://commonvoice.mozilla.org/en/datasets">Mozilla Common Voice</a> (верзија 19.0)</li>
        <li>Наставничката Валентина Степановска-Андонова од училиштето Даме Груев во Битола и нејзините ученици Ана Ванчевска, Драган Трајковски и Леона Аземовска.</li>
        <li>Учениците од Меѓународното училиште НОВА</li>
        <li>Радиолозите од болницата 8 Септември, предводени од Димитар Вељановски</li>
        <li>Дамјан Божиноски</li>
        <li>Иван Митревски</li>
        <li>Илија Глигоров</li>
        <li><a href="https://mirovnaakcija.org">Мировна Акција</a></li>
        <li><a href="https://sdk.mk/index.php/category/sakam_da_kazam/">Сакам да кажам</a></li>
        <li><a href="https://vidivaka.mk">Види Вака</a></li>
        <li><a href="https://www.tiktakaudio.com">ТикТак аудио</a></li>
    </ol>
</div>
'''

css = """
.gradio-container {
    background-color: #f3f3f3 !important;
    display: flex;
    flex-direction: column;
}
.custom-markdown p, .custom-markdown li, .custom-markdown h2, .custom-markdown a, .custom-markdown strong {
    font-size: 15px !important;
    font-family: Arial, sans-serif !important;
    color: black !important;  /* Ensure text is black */
}
button {
    color: orange !important;
}
.header {
    order: 1;
    margin-bottom: 20px;
}
.main-content {
    order: 2;
}
.footer {
    order: 3;
    margin-top: 20px;
}
.footer h2, .footer li, strong {
    color: black !important;  /* Ensure footer text is also black */
}
.header h2, .header h4, .header li, strong {
    color: black !important;  /* Ensure footer text is also black */
}
"""


transcriber_app = gr.Blocks(css=css, delete_cache=(60, 120))
    
with transcriber_app:
    state = gr.State()
    gr.HTML(project_description_header)

    gr.TabbedInterface(
        [mic_transcribe_whisper, file_transcribe_whisper],
        [" Буки-Whisper транскрипција од микрофон", "Буки-Whisper транскрипција од фајл"],
    )
    state = gr.State(value=[], delete_callback=lambda v: print("STATE DELETED"))

    gr.HTML(project_description_footer)

    transcriber_app.unload(return_prediction_whisper_mic)
    transcriber_app.unload(return_prediction_whisper_file)


# transcriber_app.launch(debug=True, share=True, ssl_verify=False)
if __name__ == "__main__":
    transcriber_app.queue()
    transcriber_app.launch(share=True)
