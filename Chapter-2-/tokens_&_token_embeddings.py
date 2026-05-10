

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