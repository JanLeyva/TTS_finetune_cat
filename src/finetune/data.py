from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import concatenate_datasets, load_dataset, Audio
import torch

def get_dataset():
    """Transform the dataset in properly format and concatenate the datasets.
    Also, transform the audio sampling to 16 Khz, the format used for the model.

    Returns:
        DatasetDict: Concatenated dataset
    """
    # "audio", "transcription" <- This will be the default columns
    dataset_openslr = load_dataset("projecte-aina/openslr-slr69-ca-trimmed-denoised")
    #  dataset_openslr "audio", "sentence", "speaker_id"
    dataset_parlament_parla = load_dataset("projecte-aina/parlament_parla", "clean")
    dataset_parlament_parla = dataset_parlament_parla.rename_column("sentence", "transcription")
    dataset_parlament_parla = dataset_parlament_parla.select_columns(["audio", "transcription"])
    # "audio", "transcription"
    dataset_festcat_trimmed_denoised = load_dataset("pprojecte-aina/festcat_trimmed_denoised")
    dataset_festcat_trimmed_denoised = dataset_festcat_trimmed_denoised.select_columns(["audio", "transcription"])
    # "audio", "sentence"
    dataset_cv_13 = load_dataset("mozilla-foundation/common_voice_13_0", "ca", split=["train", "test"])
    dataset_cv_13.rename_column("sentence", "transcription")
    dataset_cv_13 = dataset_cv_13.select_columns(["audio", "transcription"])

    # concatenate the datasets
    dataset = concatenate_datasets(
        [dataset_openslr, dataset_parlament_parla, dataset_festcat_trimmed_denoised, dataset_cv_13]
    )

    # we must re-sample to 16 Khz from 22Khz
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # transform and clean dataset
    dataset = data_transform(dataset)
    return dataset



def data_transform(dataset):
    """Data need to be transformed to be proceed"""
    replacements = [
    ('0', 'cero'),
    ( '1', 'un'),
    ( '2', 'dos'),
    ( '3', 'tres'),
    ( '4', 'cuatre'),
    ( '5', 'cinc'),
    ( '6', 'sis'),
    ( '7', 'set'),
    ( '8', 'vuit'),
    ( '9', 'nou'),
    ( 'À', 'A'),
    ( 'Á', 'A'),
    ( 'È', 'E'),
    ( 'É', 'E'),
    ( 'Í', 'I'),
    ( 'Ò', 'O'),
    ( 'Ó', 'O'),
    ( 'Ú', 'U'),
    ( 'à', 'a'),
    ( 'á', 'a'),
    ( 'ç', 's'),
    ( 'è', 'e'),
    ( 'í', 'i'),
    ( 'ï', 'i'),
    ( 'ñ', 'ny'),
    ( 'ò', 'o'),
    ( 'ó', 'o'),
    ( 'ú', 'u'),
    ( 'ü', 'u'),
    ]


    def cleanup_text(inputs):
        for src, dst in replacements:
            inputs["transcription"] = inputs["transcription"].replace(src, dst)
        return inputs

    dataset = dataset.map(cleanup_text)
    return dataset


@dataclass
class TTSDataCollatorWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_ids = [{"input_ids": feature["input_ids"]} for feature in features]
        label_features = [{"input_values": feature["labels"]} for feature in features]
        speaker_features = [feature["speaker_embeddings"] for feature in features]

        # collate the inputs and targets into a batch
        batch = processor.pad(
            input_ids=input_ids, labels=label_features, return_tensors="pt"
        )

        # replace padding with -100 to ignore loss correctly
        batch["labels"] = batch["labels"].masked_fill(
            batch.decoder_attention_mask.unsqueeze(-1).ne(1), -100
        )

        # not used during fine-tuning
        del batch["decoder_attention_mask"]

        # round down target lengths to multiple of reduction factor
        if model.config.reduction_factor > 1:
            target_lengths = torch.tensor(
                [len(feature["input_values"]) for feature in label_features]
            )
            target_lengths = target_lengths.new(
                [
                    length - length % model.config.reduction_factor
                    for length in target_lengths
                ]
            )
            max_length = max(target_lengths)
            batch["labels"] = batch["labels"][:, :max_length]

        # also add in the speaker embeddings
        batch["speaker_embeddings"] = torch.tensor(speaker_features)

        return batch
