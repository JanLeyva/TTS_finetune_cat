from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from huggingface_hub import login
from src.finetune import data, model_utils
from src.finetune.data import TTSDataCollatorWithPadding


def set_up_trainer(model, dataset_train, dataset_test, data_collator, processor):
    training_args = Seq2SeqTrainingArguments(
        output_dir="speecht5_finetuned_voxpopuli_nl",  # change to a repo name of your choice
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        warmup_steps=500,
        max_steps=4000,
        gradient_checkpointing=True,
        fp16=True,
        evaluation_strategy="steps",
        per_device_eval_batch_size=2,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        greater_is_better=False,
        label_names=["labels"],
        push_to_hub=True,
        remove_unused_columns=False, # we may need it
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        data_collator=data_collator,
        tokenizer=processor,
    )

    return trainer

if __name__ == "__main__":
    login(token="paste_here_api_key") # <- set here your hf token to upload the model into the hub
    checkpoint = "microsoft/speecht5_tts"
    # Get model, tokenizer and processor
    model = model_utils.load_model(checkpoint)
    processor, tokenizer = model_utils.load_processor_tokenizer(checkpoint)
    # Get dataset
    dataset_train, dataset_test = data.get_dataset()
    # Get data collator
    data_collator = TTSDataCollatorWithPadding(processor=processor)
    # SET UP trainner
    trainer = set_up_trainer(model, dataset_train, dataset_test, data_collator, processor)
    trainer.train()
    trainer.push_to_hub()