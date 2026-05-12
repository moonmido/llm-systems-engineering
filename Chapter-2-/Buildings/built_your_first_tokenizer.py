

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import Sequence, Lowercase
from tokenizers.processors import TemplateProcessing

"""# 1. CREATE TOKENIZER MODEL

"""

tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

"""# 2. NORMALIZATION

"""

tokenizer.normalizer = Sequence([
    Lowercase()
])

"""# 3. PRE TOKENIZATION

"""

tokenizer.pre_tokenizer= ByteLevel()

"""# 4. SPECIAL TOKENS

"""

special_tokens = [
    "<s>",
    "</s>",
    "<pad>",
    "<UNK>",
    "<mask>",
    "<code>",
    "</code>"
]

"""# 5. TRAINER

"""

trainer = BpeTrainer(
    vocab_size=30000,
    min_frequency=2,
    special_tokens=special_tokens
)

"""# 6. TRAIN

"""

files = ["code_dataset.txt"]

tokenizer.train(files, trainer)

"""# 7. POST PROCESSING

"""

tokenizer.post_processor = TemplateProcessing(
    single="<s> $A </s>",
    special_tokens=[
        ("<s>", tokenizer.token_to_id("<s>")),
        ("</s>", tokenizer.token_to_id("</s>")),
    ],
)

"""# 8. SAVE TOKENIZER

"""

tokenizer.save("tokenizer.json")

print("Tokenizer trained successfully!")

"""#9. TEST"""

tokenizer = Tokenizer.from_file("tokenizer.json")

code = """
def calculate_sum(a, b):
    return a + b
"""

encoded = tokenizer.encode("tokenizer.json")
print("TOKENS:")
print(encoded.tokens)

print("\nTOKEN IDS:")
print(encoded.ids)