# ai-bangla-
Here's a breakdown of the provided Python script:

1. **Imports:**
   - `speech_recognition` for recognizing speech.
   - `transformers` for loading the BanglaGPT model for natural language processing.

2. **Loading the BanglaGPT Model:**
   - The `AutoTokenizer` and `AutoModelForCausalLM` from the transformers library are used to load the tokenizer and model respectively.

3. **Speech Recognition Function:**
   - `recognize_speech()` function uses the `speech_recognition` library to capture and recognize Bangla speech from the microphone input.

4. **Generating Bangla Captions:**
   - `generate_bangla_caption()` function generates a Bangla caption based on the keywords provided using the BanglaGPT model.

5. **Main Execution:**
   - Captures keywords spoken by the user.
   - If keywords are recognized, they are used to generate a Bangla caption which is then printed.

Here is the complete code wrapped in a code block:

```python name=bangla_caption_generator.py
import speech_recognition as sr
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load BanglaGPT model
model_name = "csebuetnlp/banglagpt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("বাংলায় কীওয়ার্ড বলুন...")
        recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
        try:
            audio = recognizer.listen(source, timeout=5)  # Listen for 5 seconds
            text = recognizer.recognize_google(audio, language="bn-BD")  # Recognize Bangla
            return text
        except sr.UnknownValueError:
            print("দুঃখিত, আমি বুঝতে পারিনি। আবার চেষ্টা করুন।")
            return None
        except sr.RequestError:
            print("সার্ভারের সাথে সংযোগ করা যায়নি।")
            return None

def generate_bangla_caption(keywords, max_length=50):
    input_text = "ক্যাপশন তৈরি করুন: " + ", ".join(keywords)
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        early_stopping=True
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Get user input via voice
keywords_spoken = recognize_speech()

if keywords_spoken:
    keywords = keywords_spoken.strip().split(" ")  # Split words
    print(f"\nআপনার কীওয়ার্ড: {', '.join(keywords)}")

    # Generate Bangla caption
    bangla_caption = generate_bangla_caption(keywords)
    print("\nজেনারেট করা ক্যাপশন:", bangla_caption)
```

This script listens for Bangla speech input, converts it to text, and then generates a caption in Bangla based on the recognized keywords.
