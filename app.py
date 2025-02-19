import os
from flask import Flask, request, jsonify, render_template
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)

# Set your API key (you can hide this or load it from an environment variable for production)
os.environ["GROQ_API_KEY"] = 'GROQ_API_KEY'

# Define a watermarking function that inserts a single invisible Unicode character at frequent intervals
def apply_unicode_watermark(generated_text, interval=5):
    """
    Adds an invisible Unicode watermark (zero-width joiner) to the generated text at frequent intervals.
    The watermark is inserted after every 'interval' number of words.
    """
    # Unicode invisible characters (zero-width joiner)
    unicode_watermark = "\u200C"  # Single invisible watermark character
    
    words = generated_text.split()  # Split the text into words
    watermarked_text = []
    
    # Insert watermark after every 'interval' number of words
    for i in range(0, len(words), interval):
        # Add words to the list
        watermarked_text.extend(words[i:i+interval])
        # Add a single invisible watermark after each group of words
        if i + interval < len(words):  # Don't add watermark at the end if no additional words
            watermarked_text.append(unicode_watermark)
    
    # Join the words back into a string with single spaces between them
    return " ".join(watermarked_text)

# Initialize ChatGroq with your model
llm = ChatGroq(
    model="llama3-70b-8192",  # Llama3 model
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define a function to get the output and watermark it
def generate_and_watermark(user_input):
    # Define your prompt template with the user's input
    prompt = ChatPromptTemplate.from_messages([("human", user_input)])
    
    # Create a chain that combines the prompt with the LLM
    chain = LLMChain(prompt=prompt, llm=llm)
    
    # Generate output using LangChain
    generated_text = chain.run(input=user_input)

    # Apply the Unicode watermark (single zero-width joiner) to the generated text
    watermarked_text = apply_unicode_watermark(generated_text)

    return watermarked_text

@app.route('/')
def home():
    return render_template('index.html')  # Default generation page

@app.route('/watermark-detector')
def watermark_detector():
    return render_template('detect_watermark.html')  # Watermark detection page


@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    prompt = data.get("prompt", "")

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Generate text and apply watermark
        watermarked_text = generate_and_watermark(prompt)
    except Exception as e:
        return jsonify({"error": f"Text generation failed: {str(e)}"}), 500

    return jsonify({
        "watermarked_text": watermarked_text
    })


@app.route('/detect', methods=['POST'])
def detect_watermark():
    data = request.get_json()  # Receive JSON data from the client
    text = data.get("text", "")
    
    # Check for the presence of the Unicode watermark (\u200C)
    if "\u200C" in text:
        result = {"message": "This text is AI-generated."}
    else:
        result = {"message": "No AI watermark detected in this text."}

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
