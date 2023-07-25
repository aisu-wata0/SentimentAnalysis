
from dataclasses import dataclass
import torch
import numpy as np

from transformers import BertTokenizer
from transformers import DistilBertTokenizer

from .model import BertForMultiLabelClassification

def restructure_result(result):
	emotion_scores = {}
	for i in range(len(result['labels'])):
		emotion_scores[result['labels'][i]] = result['scores'][i]
	return emotion_scores

model_cache = {}
tokenizer_cache = {}

@dataclass
class SentimentClassifier:
	# 8 emotions
	model_name: str = "j-hartmann/emotion-english-distilroberta-base"
	# model_name: str = "nateraw/bert-base-uncased-emotion"
	# model_name: str = "emtract-distilbert-base-uncased-emotion"
	# 28 emotions
	# model_name: str = "bhadresh-savani/bert-base-go-emotion"
	# model_name: str = "joeddav/distilbert-base-uncased-go-emotions-student"
	tokenizer_name: str | None = None # Default same as model's name
	model_type: type = BertForMultiLabelClassification
	tokenizer_type: type = DistilBertTokenizer

	def __post_init__(self):
		if not self.tokenizer_name:
			self.tokenizer_name = self.model_name
		
		if self.tokenizer_name not in tokenizer_cache:
			tokenizer_cache[self.tokenizer_name] = self.tokenizer_type.from_pretrained(self.tokenizer_name)
		self.tokenizer = tokenizer_cache[self.tokenizer_name]

		if self.model_name not in model_cache:
			model_cache[self.model_name] = self.model_type.from_pretrained(self.model_name)
		self.model = model_cache[self.model_name]

	def get_results(self, texts):
		results = []
		for txt in texts:
			inputs = self.tokenizer(txt, return_tensors="pt")
			outputs = self.model(**inputs)
			scores =  1 / (1 + torch.exp(-outputs[0]))  # Sigmoid
			threshold = .0
			for item in scores:
				labels = []
				scores = []
				for idx, s in enumerate(item):
					if s > threshold:
						labels.append(self.model.config.id2label[idx])
						scores.append(float(s))
				results.append({"labels": labels, "scores": scores})
		return results

	def get(self, text):
		return restructure_result(self.get_results([text])[0])

	def get_list(self, texts):
		return [restructure_result(r) for r in self.get_results(texts)]