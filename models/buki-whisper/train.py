#!/usr/bin/env/python3

import sys
import os

import torch
from torch.utils.data import DataLoader
import torchaudio
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.data_utils import undo_padding
from speechbrain.utils.distributed import if_main_process, run_on_main
import logging
from transformers import AutoTokenizer

from jiwer import wer, cer

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        bos_tokens, bos_tokens_lens = batch.tokens_bos

        if stage == sb.Stage.TRAIN:
            wavs, self.wav_lens = self.hparams.wav_augment(wavs, wav_lens)

        # We compute the padding mask and replace the values with the pad_token_id
        # that the Whisper decoder expect to see.
        abs_tokens_lens = (bos_tokens_lens * bos_tokens.shape[1]).long()
        pad_mask = (torch.arange(abs_tokens_lens.max(), device=self.device)[None, :] < abs_tokens_lens[:, None])
        bos_tokens[~pad_mask] = self.tokenizer.pad_token_id

        # Forward encoder + decoder
        enc_out, logits, _ = self.modules.whisper(wavs, bos_tokens)
        log_probs = self.hparams.log_softmax(logits)

        hyps = None
        if stage == sb.Stage.VALID:
            hyps, _, _, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _, _, _ = self.hparams.test_search(enc_out.detach(), wav_lens)

        return log_probs, hyps, wav_lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss NLL given predictions and targets."""

        (log_probs, hyps, wav_lens) = predictions
        batch = batch.to(self.device)
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos

        # Augment Labels
        # if stage == sb.Stage.TRAIN and hasattr(self.hparams, "wav_augment"):
        #     tokens_eos = self.hparams.wav_augment.replicate_labels(tokens_eos)
        #     tokens_eos_lens = self.hparams.wav_augment.replicate_labels(
        #         tokens_eos_lens
        #     )

        loss = self.hparams.nll_loss(log_probs, tokens_eos, length=tokens_eos_lens)

        if stage != sb.Stage.TRAIN:
            tokens, tokens_lens = batch.tokens

            # Decode token terms to words
            predicted_words = [self.tokenizer.decode(t, skip_special_tokens=True).strip() for t in hyps]

            # Convert indices to words
            target_words = undo_padding(tokens, tokens_lens)
            target_words = self.tokenizer.batch_decode(target_words, skip_special_tokens=True)

            if hasattr(self.hparams, "normalized_transcripts"):
                predicted_words = [self.tokenizer.normalize(text).split(" ") for text in predicted_words]
                target_words = [self.tokenizer.normalize(text).split(" ") for text in target_words]
            else:
                predicted_words = [text.split(" ") for text in predicted_words]
                target_words = [text.split(" ") for text in target_words]

            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            lr = self.hparams.lr_annealing_whisper.current_lr
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"WER": stage_stats["WER"]},
                min_keys=["WER"],
            )
        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            if if_main_process():
                with open(self.hparams.test_wer_file, "w") as w:
                    self.wer_metric.write_stats(w)

    def run_inference(
            self,
            dataset, # Must be obtained from the dataio_function
            min_key, # We load the model with the lowest error rate
            loader_kwargs, # opts for the dataloading
        ):

        # If dataset isn't a Dataloader, we create it. 
        if not isinstance(dataset, DataLoader):
            loader_kwargs["ckpt_prefix"] = None
            dataset = self.make_dataloader(
                dataset, sb.Stage.TEST, **loader_kwargs
            )

        self.checkpointer.recover_if_possible(min_key=min_key)
        self.modules.eval() # We set the model to eval mode (remove dropout etc)

        with torch.no_grad():
            true_labels = []
            pred_labels = []
            #for batch in tqdm(dataset, dynamic_ncols=True):

            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 

                tokens, tokens_lens = batch.tokens
                log_probs, predictions, wav_lens = self.compute_forward(batch, stage=sb.Stage.TEST) 
                pred_batch = []
                predicted_words = []

                # Decode token terms to words
                predicted_words = [tokenizer.decode(token, skip_special_tokens=True).strip() for token in predictions]
                # predicted_words = [tokenizer.decode(pred) for pred in predictions]
                # labels = [tokenizer.decode(trn) for trn in batch.tokens_list]

                # Convert indices to words
                target_words = undo_padding(tokens, tokens_lens)
                target_words = tokenizer.batch_decode(target_words, skip_special_tokens=True)

                # if hasattr(self.hparams, "normalized_transcripts"):
                #     predicted_words = [tokenizer.normalize(text) for text in predicted_words]
                #     target_words = [tokenizer.normalize(text) for text in target_words]

                for sent in predicted_words:
                    sent = filter_repetitions([sent], 3)
                    sent = " ".join(sent)
                    pred_batch.append(sent)

                # if len(pred_batch[0].split()) > 50:
                    # continue
                pred_labels.append(pred_batch[0])
                true_labels.append(target_words[0])

                # print("True: ", batch.transcript[0])
                # print("Pred: ", pred_batch[0])
                # with open("predictions/predictions_arhiv.txt", "a") as f:
                    # f.write("True: " + batch.transcript[0] + "\n")
                    # f.write("Pred: " + pred_batch[0] + "\n\n")

                if self.hparams.restore_capitalization:
                    inputs = recap_tokenizer(["restore capitalization and punctuation: " + pred_batch[0]], return_tensors="pt", padding=True).to(self.device)
                    outputs = recap_model.generate(**inputs, max_length=1024, num_beams=5, early_stopping=True).squeeze(0)
                    pred_batch[0] = recap_tokenizer.decode(outputs, skip_special_tokens=True)
                

                # print("True: ", target_words[0])
                # print("Pred: ", pred_batch[0])
                # print('WER: ', wer(target_words, pred_batch[0]) * 100)
                # print("\n")

                # with open("predictions/predictions_eaz.txt", "a") as f:
                #     f.write(str(batch.id[0]) + "\t" + pred_batch[0] + "\n")

        print('WER: ', wer(true_labels, pred_labels) * 100)
        print('CER: ', cer(true_labels, pred_labels) * 100)


def filter_repetitions(seq, max_repetition_length):
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


def dataio_prepare(hparams, tokenizer):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(hparams["data_folder"], "train_dev.json"), replacements={"data_root": data_folder})
    train_data = train_data.filtered_sorted(sort_key="duration")
    hparams["train_dataloader_opts"]["shuffle"] = False

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(hparams["data_folder"], "test_all.json"), replacements={"data_root": data_folder})
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(hparams["data_folder"], "test_eaz.json"), replacements={"data_root": data_folder})

    datasets = [train_data, valid_data, test_data]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("data_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(data_path):
        info = torchaudio.info(data_path)
        sig = sb.dataio.dataio.read_audio(data_path)
        if info.sample_rate != hparams["sample_rate"]:
            sig = torchaudio.transforms.Resample(info.sample_rate, hparams["sample_rate"])(sig)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(transcript):
        # if hasattr(hparams, "normalized_transcripts"):
        #     transcript = tokenizer.normalize(transcript)
        yield transcript
        tokens_list = tokenizer.encode(transcript, add_special_tokens=False)
        yield tokens_list
        tokens_list = tokenizer.build_inputs_with_special_tokens(tokens_list)
        tokens_bos = torch.LongTensor(tokens_list[:-1])
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list[1:])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "sig", "tokens_list", "tokens_bos", "tokens_eos", "tokens"],
    )

    return train_data, valid_data, test_data


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Defining tokenizer and loading it
    tokenizer = hparams["whisper"].tokenizer

    # here we create the datasets objects as well as tokenization and encoding
    train_data, valid_data, test_data = dataio_prepare(hparams, tokenizer)

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
        opt_class=hparams["whisper_opt_class"],
    )

    # We load the pretrained whisper model
    if "pretrainer" in hparams.keys():
        hparams["pretrainer"].collect_files()
        hparams["pretrainer"].load_collected(asr_brain.device)

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for Whisper.
    asr_brain.tokenizer = tokenizer


    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        # Training
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_opts"],
            valid_loader_kwargs=hparams["valid_dataloader_opts"],
        )
    
    else:
        # evaluate
        print("Evaluating")
        asr_brain.run_inference(test_data, "WER", hparams["test_dataloader_opts"])

