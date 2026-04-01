with open("/content/the-verdict.txt", "r", encoding="utf-8") as f:
  raw_ds = f.read()

print(len(raw_ds))
print(raw_ds[:99])

####################################################

#we can do this but we will be doing BPE insted
#this is fro  the tokenizer code | just change the var_names

# class SimpleTokenizerV2:
#   def __init__(self, vocab):
#     self.str_to_int = vocab
#     self.int_to_str = {i:s for s, i in vocab.items()}

#   def encode(self, text):
#     preprocess = re.split(r'([.,:;?!()"\']|--|\s)', text)
#     preprocess = [
#         item.strip() for item in preprocess if item.strip()
#     ]
#     #this is the extra line or change we have do rest is same

#     preprocess = [
#         item if item in self.str_to_int
#           else "<|unk|>" for item in preprocess
#     ]
#     ids = [self.str_to_int[s] for s in preprocess]
#     return ids

#   def decode(self, ids):
#     text = " ".join([self.int_to_str[i] for i in ids])
#     text = re.sub(r'\s+([.,:;?!()"\'])', r'\1', text)
#     return text
####################################################
import importlib
import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')

encoded_text = tokenizer.encode(raw_ds)
print(len(encoded_text))

enc_sample = encoded_text[50:]
#####################################################
context_size = 4
#basically the input x is the first 4 tokens, and the target y 
#is the next 4 tokens 

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x: {x}")
print(f"y:      {y}")
