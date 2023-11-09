import torch.nn as nn

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    BertForNextSentencePrediction,
    BertTokenizer,
    BertTokenizerFast,
    GPT2LMHeadModel
)


class BaseModel(nn.Module):
    def __init__(self, model_path, max_len=None, padding=True, truncation=True):
        super(BaseModel, self).__init__()

        self.model_config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = None

        if hasattr(self.model_config, 'tokenizer_class'):
            tokenizer_name = self.model_config.tokenizer_class
            if tokenizer_name == 'BertTokenizer':
                self.tokenizer = BertTokenizer.from_pretrained(model_path)
            elif tokenizer_name == 'BertTokenizerFast':
                self.tokenizer = BertTokenizerFast.from_pretrained(model_path)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        if max_len is None:
            if hasattr(self.model_config, 'max_position_embeddings'):
                max_len = self.model_config.max_position_embeddings
            elif hasattr(self.model_config, 'n_positions'):
                max_len = self.model_config.n_positions
            else:
                max_len = 512

        self.max_len = max_len

        self.padding = padding
        self.truncation = truncation

    def tokenize(self, text):
        text_ids = self.tokenizer(text=text,
                                  add_special_tokens=True,
                                  padding=self.padding,
                                  truncation=self.truncation,
                                  max_length=self.max_len,
                                  return_tensors="pt")
        return text_ids

    def forward(self, inputs):
        raise NotImplementedError("Please Implement this method")


class NSPModel(BaseModel):
    def __init__(self, model_path, max_len=None, padding=True, truncation=True):
        BaseModel.__init__(self,
                           model_path=model_path,
                           max_len=max_len,
                           padding=padding,
                           truncation=truncation)

        self.transformer = BertForNextSentencePrediction.from_pretrained(model_path)

    def forward(self, inputs):
        return self.transformer(**inputs)


class NormalModel(BaseModel):
    def __init__(self, model_path, max_len=None, padding=True, truncation=True):
        BaseModel.__init__(self,
                           model_path=model_path,
                           max_len=max_len,
                           padding=padding,
                           truncation=truncation)

        self.transformer = AutoModel.from_pretrained(model_path)

    def forward(self, inputs):
        return self.transformer(**inputs)


class SentenceModel(BaseModel):
    def __init__(self, model_path, max_len=None, padding=True, truncation=True):
        BaseModel.__init__(self,
                           model_path=model_path,
                           max_len=max_len,
                           padding=padding,
                           truncation=truncation)

        self.transformer = SentenceTransformer(model_path)

    def forward(self, inputs):
        return self.transformer.encode(inputs, convert_to_tensor=True)


class GPTModel(BaseModel):
    def __init__(self, model_path, max_len=None, padding=True, truncation=True):
        BaseModel.__init__(self,
                           model_path=model_path,
                           max_len=max_len,
                           padding=padding,
                           truncation=truncation)

        self.transformer = GPT2LMHeadModel.from_pretrained(model_path)

    def forward(self, inputs):
        outputs = self.transformer(**inputs, labels=inputs["input_ids"])

        return outputs
