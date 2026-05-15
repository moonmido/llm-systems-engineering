
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=False
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

prompt = "Write an email apologizing to Sarah for the tragic gardening mishap. Explain how it happened.<|assistant|>"
input_ids = tokenizer(prompt,return_tensors="pt").input_ids.to("cuda")

generation_output = model.generate(
    input_ids=input_ids,
    max_new_tokens=20
)
print(tokenizer.decode(generation_output[0]))

print(input_ids)

for id in input_ids[0]:
  print(tokenizer.decode(id))

generation_output

print(tokenizer.decode(3323))
print(tokenizer.decode(622))
print(tokenizer.decode([3323, 622]))
print(tokenizer.decode(29901))

"""### **Comparing Trained LLM Tokenizers**"""

colors_list = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence,tokenizer_name):
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  token_ids = tokenizer(sentence).input_ids
  for idx, t in enumerate(token_ids):
            print(
            f'\x1b[0;30;48;2;{colors_list[idx % len(colors_list)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )

text = """
English and CAPITALIZATION
🎵 鸟
show_tokens False None elif == >= else: two tabs:"    " Three tabs: "       "
12.0*50=600
"""

show_tokens(text, "bert-base-uncased")

show_tokens(text, "bert-base-cased")

show_tokens(text, "gpt2")

show_tokens(text, "google/flan-t5-small")

show_tokens(text, "Xenova/gpt-4")

show_tokens(text, "bigcode/starcoder2-15b")

show_tokens(text, "facebook/galactica-1.3b")

show_tokens(text, "microsoft/Phi-3-mini-4k-instruct")

"""## **Contextualized Word Embeddings From a Language Model (Like BERT)**"""

from transformers import AutoModel,AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-base")
model = AutoModel.from_pretrained("microsoft/deberta-v3-xsmall")
tokens = tokenizer('Hello World',return_tensors='pt')
output=model(**tokens)[0]

output.shape

for token in tokens['input_ids'][0]:
  print(tokenizer.decode(token))

output

"""## **Text Embeddings (For Sentences and Whole Documents)**"""

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
vector = model.encode("Best movie ever !")

vector.shape

"""### **Word Embeddings Beyond LLMs**"""

!pip install gensim

import gensim.downloader as api

model = api.load("glove-wiki-gigaword-50")

model.most_similar([model['king']], topn=20)