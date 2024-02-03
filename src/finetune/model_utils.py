# load processor and model
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech
from functools import partial

def load_processor_tokenizer(checkpoint):
	processor = SpeechT5Processor.from_pretrained(checkpoint)
	tokenizer = processor.tokenizer
	return processor, tokenizer

def load_model(checkpoint):
	model = SpeechT5ForTextToSpeech.from_pretrained(checkpoint)
	# disable cache during training since it's incompatible with gradient checkpointing
	model.config.use_cache = False

	# set language and task for generation and re-enable cache
	model.generate = partial(model.generate, use_cache=True)
	return model
