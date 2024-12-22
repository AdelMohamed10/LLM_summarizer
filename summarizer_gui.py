pip install gradio
import gradio as gr
import torch
from transformers import T5ForConditionalGeneration, AutoTokenizer, MarianMTModel, MarianTokenizer

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# best parameters
BEST_PARAMS = {
    'learning_rate': 9.184880850544312e-05,
    'batch_size': 8,
    'warmup_steps': 187,
    'num_beams': 3
}

# Load T5 tokenizer and model (direct initialization)
tokenizer_t5 = AutoTokenizer.from_pretrained("t5-base")
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-base")
model_t5.to(device)

# Load translation model (English to Arabic)
model_name_hein = "Helsinki-NLP/opus-mt-en-ar"
model_hein = MarianMTModel.from_pretrained(model_name_hein)
tokenizer_hein = MarianTokenizer.from_pretrained(model_name_hein)
model_hein.to(device)

# Function for summarization
def summarize(text):
    inputs = tokenizer_t5(
        "summarize: " + text,
        return_tensors="pt",
        max_length=1024,  # Maximum input length
        truncation=True,
    ).to(device)

    outputs = model_t5.generate(
        inputs["input_ids"],
        max_length=700,  # Increase this value for longer summaries
        min_length=200,  # Optional: Set a minimum length for the summary
        num_beams=BEST_PARAMS['num_beams'],  # Use best parameter
        length_penalty=2.0,  # Encourage longer sequences
        early_stopping=True
    )

    return tokenizer_t5.decode(outputs[0], skip_special_tokens=True)

# Function for translation to Arabic
def translate_to_arabic(text):
    inputs = tokenizer_hein(text, return_tensors="pt", padding=True, truncation=True).to(device)
    outputs = model_hein.generate(inputs["input_ids"], max_length=512)
    return tokenizer_hein.decode(outputs[0], skip_special_tokens=True)

# Function for text statistics
def text_statistics(text):
    word_count = len(text.split())
    sentence_count = text.count(".")
    return f"Word Count: {word_count}, \nSentence Count: {sentence_count}"

# Gradio UI
def process_request(task, text):
    """Process the user's request based on the selected task."""
    if task == "Summarize":
        return summarize(text)
    elif task == "Translate to Arabic":
        return translate_to_arabic(text)
    elif task == "Text Statistics":
        return text_statistics(text)
    else:
        return "Invalid task selected."

# Gradio UI with task selection
interface = gr.Interface(
    fn=process_request,
    inputs=[
        gr.Radio(["Summarize", "Translate to Arabic", "Text Statistics"], label="Task"),
        gr.Textbox(lines=10, placeholder="Enter text here...", label="Input Text"),
    ],
    outputs="text",
    title="Research Assistant with Summarization and Translation",
    description="Choose a task to perform: summarization, translation, or text analysis.",
)

if __name__ == "__main__":
    interface.launch()
