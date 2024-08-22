from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer
import os
# Set your Hugging Face API token
token = "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"

# Authenticate using your token
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", revision="main", auth_token=token)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", revision="main", auth_token=token)

# Function to generate a joke
def generate_joke():
    # Prompt for generating a joke
    prompt = "Tell me a joke:"
    
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate joke using the model
    joke_ids = model.generate(inputs.input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2, repetition_penalty=1.5, top_p=0.92, temperature=.85)
    
    # Decode generated joke
    generated_joke = tokenizer.decode(joke_ids[0], skip_special_tokens=True)
    
    return generated_joke

# Example usage:
generated_joke = generate_joke()
print("Generated Joke:", generated_joke)
