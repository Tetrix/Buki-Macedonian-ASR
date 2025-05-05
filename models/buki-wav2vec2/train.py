#!/usr/bin/env/python3

import logging
import sys
from pathlib import Path
import os

import librosa

import torch
from torch.utils.data import DataLoader
from hyperpyyaml import load_hyperpyyaml

import speechbrain as sb
from speechbrain.utils.distributed import if_main_process, run_on_main

from jiwer import wer, cer

logger = logging.getLogger(__name__)


# Define training procedure
class ASR(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        sig, self.sig_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos
        sig, self.sig_lens = sig.to(self.device), self.sig_lens.to(self.device)

        # Add waveform augmentation if specified.
        if stage == sb.Stage.TRAIN:
            sig, self.sig_lens = self.hparams.wav_augment(sig, self.sig_lens)

        # Forward pass
        encoded_outputs = self.modules.encoder_w2v2(sig.detach())
        embedded_tokens = self.modules.embedding(tokens_bos)
        decoder_outputs, _ = self.modules.decoder(embedded_tokens, encoded_outputs, self.sig_lens)

        # Output layer for seq2seq log-probabilities
        logits = self.modules.seq_lin(decoder_outputs)
        predictions = {"seq_logprobs": self.hparams.log_softmax(logits)}
       
        if self.is_ctc_active(stage):
            # Output layer for ctc log-probabilities
            ctc_logits = self.modules.ctc_lin(encoded_outputs)
            predictions["ctc_logprobs"] = self.hparams.log_softmax(ctc_logits)
        elif stage == sb.Stage.VALID:
            predictions["tokens"], _, _, _ = self.hparams.greedy_search(encoded_outputs, self.sig_lens)
        elif stage == sb.Stage.TEST:
            predictions["tokens"], _, _, _ = self.hparams.test_search(encoded_outputs, self.sig_lens)

        return predictions
    

    def is_ctc_active(self, stage):
        """Check if CTC is currently active.
        Arguments
        ---------
        stage : sb.Stage
            Currently executing stage.
        """
        if stage != sb.Stage.TRAIN:
            return False
        current_epoch = self.hparams.epoch_counter.current
        return current_epoch <= self.hparams.number_of_ctc_epochs



    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss = self.hparams.nll_cost(log_probabilities=predictions["seq_logprobs"], targets=tokens_eos, length=tokens_eos_lens)

        if self.is_ctc_active(stage):
            # Load tokens without EOS as CTC targets
            loss_ctc = self.hparams.ctc_cost(predictions["ctc_logprobs"], tokens, self.sig_lens, tokens_lens)
            loss *= 1 - self.hparams.ctc_weight
            loss += self.hparams.ctc_weight * loss_ctc

        if stage != sb.Stage.TRAIN:
            predicted_words = [self.hparams.tokenizer.decode_ids(prediction).split(" ") for prediction in predictions["tokens"]]
            target_words = [words.split(" ") for words in batch.transcript]
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_words, target_words)

        return loss

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.cer_metric = self.hparams.cer_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(stage_stats["WER"])
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)
            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
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
            for batch in dataset:
                # Make sure that your compute_forward returns the predictions !!!
                # In the case of the template, when stage = TEST, a beam search is applied 
                # in compute_forward(). 
                predictions = self.compute_forward(batch, stage=sb.Stage.TEST) 

                pred_batch = []
                predicted_words = []

                predicted_words = [self.hparams.tokenizer.decode_ids(prediction).split(" ") for prediction in predictions["tokens"]]
                for sent in predicted_words:
                    # sent = " ".join(sent)
                    sent = filter_repetitions(sent, 3)
                    sent = " ".join(sent)
                    pred_batch.append(sent)

                pred_labels.append(pred_batch[0])
                true_labels.append(batch.transcript[0])

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

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    """
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(hparams["data_folder"], "train.json"), replacements={"data_root": data_folder})
    train_data = train_data.filtered_sorted(sort_key="duration")
    hparams["train_dataloader_opts"]["shuffle"] = False

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(hparams["data_folder"], "dev.json"), replacements={"data_root": data_folder})
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_json(json_path=os.path.join(hparams["data_folder"], "test.json"), replacements={"data_root": data_folder})


    datasets = [train_data, valid_data, test_data]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("data_path")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(data_path):
        sig, sr = librosa.load(data_path, sr=16000)
        # sig = sb.dataio.dataio.read_audio(wav) # alternatively use the SpeechBrain data loading function
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("transcript")
    @sb.utils.data_pipeline.provides("transcript", "tokens_list", "tokens_bos", "tokens_eos", "tokens")
    def text_pipeline(transcript):
        yield transcript
        tokens_list = tokenizer.encode_as_ids(transcript)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(datasets, ["id", "sig", "transcript", "tokens_list", "tokens_bos", "tokens_eos", "tokens"])

    return (train_data, valid_data, test_data)


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

    # here we create the datasets objects as well as tokenization and encoding
    (train_data, valid_data, test_data) = dataio_prepare(hparams)

    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected()

    # Trainer initialization
    asr_brain = ASR(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # We dynamically add the tokenizer to our brain class.
    # NB: This tokenizer corresponds to the one used for the LM!!
    asr_brain.tokenizer = hparams["tokenizer"]
    train_dataloader_opts = hparams["train_dataloader_opts"]
    valid_dataloader_opts = hparams["valid_dataloader_opts"]


    # Training/validation loop
    if hparams["skip_training"] == False:
        print("Training...")
        # Training
        asr_brain.fit(
            asr_brain.hparams.epoch_counter,
            train_data,
            valid_data,
            train_loader_kwargs=train_dataloader_opts,
            valid_loader_kwargs=valid_dataloader_opts,
        )
    
    else:
        # evaluate
        print("Evaluating")
        asr_brain.run_inference(test_data, "WER", hparams["test_dataloader_opts"])
