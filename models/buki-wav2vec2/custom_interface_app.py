import torch
from speechbrain.inference.interfaces import Pretrained
import librosa
import numpy as np
import torchaudio
import os

class ASR(Pretrained):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode_batch_w2v2(self, device, wavs, wav_lens=None, normalize=False):
        wavs = wavs.to(device)
        wav_lens = wav_lens.to(device)

        # Forward pass
        encoded_outputs = self.mods.encoder_w2v2(wavs.detach())
        # append
        tokens_bos = torch.zeros((wavs.size(0), 1), dtype=torch.long).to(device)
        embedded_tokens = self.mods.embedding(tokens_bos)
        decoder_outputs, _ = self.mods.decoder(embedded_tokens, encoded_outputs, wav_lens)

        # Output layer for seq2seq log-probabilities
        predictions = self.hparams.test_search(encoded_outputs, wav_lens)[0]
        # predicted_words = [self.hparams.tokenizer.decode_ids(prediction).split(" ") for prediction in predictions]
        predicted_words = []
        for prediction in predictions:
            prediction = [token for token in prediction if token != 0]
            predicted_words.append(self.hparams.tokenizer.decode_ids(prediction).split(" "))
        prediction = []
        for sent in predicted_words:
            sent = self.filter_repetitions(sent, 3)
            prediction.append(sent)
        predicted_words = prediction
        return predicted_words


    def filter_repetitions(self, seq, max_repetition_length):
        seq = list(seq)
        output = []
        max_n = len(seq) // 2
        for n in range(max_n, 0, -1):
            max_repetitions = max(max_repetition_length // n, 1)
            # Don't need to iterate over impossible n values:
            # len(seq) can change a lot during iteration
            if (len(seq) <= n*2) or (len(seq) <= max_repetition_length):
                continue
            iterator = enumerate(seq)
            # Fill first buffers:
            buffers = [[next(iterator)[1]] for _ in range(n)]
            for seq_index, token in iterator:
                current_buffer = seq_index % n
                if token != buffers[current_buffer][-1]:
                    # No repeat, we can flush some tokens
                    buf_len = sum(map(len, buffers))
                    flush_start = (current_buffer-buf_len) % n
                    # Keep n-1 tokens, but possibly mark some for removal
                    for flush_index in range(buf_len - buf_len%n):
                        if (buf_len - flush_index) > n-1:
                            to_flush = buffers[(flush_index + flush_start) % n].pop(0)
                        else:
                            to_flush = None
                        # Here, repetitions get removed:
                        if (flush_index // n < max_repetitions) and to_flush is not None:
                            output.append(to_flush)
                        elif (flush_index // n >= max_repetitions) and to_flush is None:
                            output.append(to_flush)
                buffers[current_buffer].append(token)
            # At the end, final flush
            current_buffer += 1
            buf_len = sum(map(len, buffers))
            flush_start = (current_buffer-buf_len) % n
            for flush_index in range(buf_len):
                to_flush = buffers[(flush_index + flush_start) % n].pop(0)
                # Here, repetitions just get removed:
                if flush_index // n < max_repetitions:
                    output.append(to_flush)
            seq = []
            to_delete = 0
            for token in output:
                if token is None:
                    to_delete += 1
                elif to_delete > 0:
                    to_delete -= 1
                else:
                    seq.append(token)
            output = []
        return seq


    def classify_file_w2v2(self, file, vad_model, device):
        # Get audio length in seconds
        sr = 16000
        max_segment_length = 30

        waveform, file_sr = torchaudio.load(file)
        waveform = waveform.mean(dim=0, keepdim=True) # convert to mono
        # resample if not 16kHz
        if file_sr != sr:
            waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)

        # limit to 1 min
        # waveform = waveform[:, :60*sr]
        
        waveform = waveform.squeeze()
        audio_length = len(waveform) / sr
        print(f"Audio length: {audio_length:.2f} seconds")
        
        if audio_length >= max_segment_length:
            print(f"Audio is too long ({audio_length:.2f} seconds), splitting into segments")
            
            # save waveform temporarily
            torchaudio.save("temp.wav", waveform.unsqueeze(0), sr)
            # get boundaries based on VAD
            boundaries = vad_model.get_speech_segments("temp.wav", 
                                                large_chunk_size=30, 
                                                small_chunk_size=10, 
                                                apply_energy_VAD=True,
                                                double_check=True)
            # remove temp file
            os.remove("temp.wav")

            # Merge the segments to max max_segment_length
            segments = []
            current_start = boundaries[0][0].item()
            current_end = boundaries[0][1].item()

            for i in range(1, len(boundaries)):
                next_start = boundaries[i][0].item()
                next_end = boundaries[i][1].item()

                # Check if the current segment can merge with the next segment
                if (current_end - current_start) + (next_end - next_start) <= max_segment_length:
                    # Extend the current segment
                    current_end = next_end
                else:
                    # Add the current segment to the result and start a new one
                    segments.append([current_start, current_end])
                    current_start = next_start
                    current_end = next_end

            # Add the last segment
            segments.append([current_start, current_end])

            # Process each segment
            outputs = []
            for i, segment in enumerate(segments):
                start, end = segment
                start = int(start * sr)
                end = int(end * sr)
                segment = waveform[start:end]
                print(f"Processing segment {i + 1}/{len(segments)}, length: {len(segment) / sr:.2f} seconds")

                # import soundfile as sf
                # sf.write(f"outputs/segment_{i}.wav", segment, sr)

                segment_tensor = torch.tensor(segment).to(device)

                # Fake a batch for the segment
                batch = segment_tensor.unsqueeze(0).to(device)
                rel_length = torch.tensor([1.0]).to(device)  # Adjust if necessary

                # Pass the segment through the ASR model
                result = " ".join(self.encode_batch_w2v2(device, batch, rel_length)[0])
                # outputs.append(result)
                yield result
        else:
            waveform, file_sr = torchaudio.load(file)
            # resample if not 16kHz
            if file_sr != sr:
                waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)
            waveform = waveform.to(device)
            rel_length = torch.tensor([1.0]).to(device)
            outputs = " ".join(self.encode_batch_w2v2(device, waveform, rel_length)[0])
            yield outputs

