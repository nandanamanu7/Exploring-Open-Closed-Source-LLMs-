from transformers import pipeline

# --- Task 1:

generator = pipeline("text-generation", model="gpt2")

# -- initial prompt

# prompts model
prompt = "Explain how to create a Machine Learning Model in simple terms"
# defines starting sentence, max output length, # of answers
output = generator(prompt, max_length=100, num_return_sequences=1,
# controls creativity (0.7 is medium), next word chosen from top 50 words
temperature = 0.7, top_k=50)
print(output[0]['generated_text'])

# -- prompt modification
# Casual Prompt

prompt = "What is a transformer model?"
output = generator(prompt, max_length=100, num_return_sequences=1, temperature = 0.7, top_k=50)
print(output[0]['generated_text'])

# Detailed Prompt

prompt = "Explain how to create a Machine Learning Model in simple terms for a 10 year old child."
output = generator(prompt, max_length=100, num_return_sequences=1, temperature = 0.7, top_k=50)
print(output[0]['generated_text'])

# Structured Prompt

prompt = "Give me a step-by-step guide on how to build a simple transformer model in 2 hours"
output = generator(prompt, max_length=100, num_return_sequences=1, temperature = 0.7, top_k=50)
print(output[0]['generated_text'])

# --- Task 2

import os
from huggingface_hub import InferenceClient

MODEL_ID = "tiiuae/falcon-7b-instruct"
# Best practice: store your token in an env var like HF_TOKEN
# export HF_TOKEN="hf_..."
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_API_KEY")
client = InferenceClient(model=MODEL_ID, token=HF_TOKEN)
prompt = "Explain Artificial Intelligence in simple terms."
# Text-generation API (works for instruct-style causal LMs)
result = client.text_generation(
prompt,
max_new_tokens=150,
temperature=0.7,
top_p=0.9,
)
print(result)