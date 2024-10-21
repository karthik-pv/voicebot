from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
from utils import fetch_product_info

app = Flask(__name__)

conversation_history = []

# Initialize the model globally
GOOGLE_API_KEY = "AIzaSyCadPuPUQvtH-NsETbzmgooO9OT2NkAt1s"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Global variables for fetched data and previous question/answer
fetched_data = None
prevQ = ""
prevA = ""


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(
            model=model, content=input, task_type="retrieval_document", title=title
        )["embedding"]


def create_chroma_db(documents, name):
    chroma_client = chromadb.Client(tenant="default_tenant")
    db = chroma_client.create_collection(
        name=name, embedding_function=GeminiEmbeddingFunction()
    )
    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))
    return db


def get_relevant_passage(query, db):
    results = db.query(query_texts=[query], n_results=7)
    passages = results["documents"][0]
    return passages


def make_prompt(query, relevant_passage, prevQ, prevA):
    prompt = (
        """You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, and don’t include all relevant background information, just the bare minimum. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and conversational tone. \
  If the passage is irrelevant to the answer, you may ignore it. \
  If the data is not sufficient, just return the response text as 'Data Insufficient'....nothing else. \
  The response must be short....around 75-100 words. \
  Answer the question as if you are a human salesman......end every response with a question to help the user further.....example volunteer extra information.....volunteer to compare the products and provide the best. \
  QUESTION: '{query}' \
  PASSAGE: '{relevant_passage}' \
  PREVIOUS QUESTION: '{prevQ}' \
  PREVIOUS ANSWER: '{prevA}' \
  Look at the previous question and answer.....and answer in such a way that the context is maintained......I want the user to feel like it is a single conversation goin on and not individual statements......use the contenxt in the previous question and answer \
  ANSWER:
  """
    ).format(query=query, relevant_passage=relevant_passage, prevA=prevA, prevQ=prevQ)

    return prompt


def identify_topic(query):
    prompt = (
        "From the given sentence identify the topic of discussion.......the topic will mostly always be a product......give me the \
    range of products the user is looking for, e.g., if the search query is 'how good is samsung m32', you need to return 'samsung m32'......in max of \
    3 words just tell me the topic.....i want your response to be as accurate as possible. \
    Sentence: '{query}'"
    ).format(query=query)

    topic = model.generate_content(prompt)
    try:
        text = topic._result.candidates[0].content.parts[0].text
        return text
    except AttributeError:
        return None


def text_to_speech(text):
    tts = gTTS(text=text, lang="en-us", tld="us", slow=False)
    filename = "response.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone(sample_rate=16000, chunk_size=1024) as source:
        print("Listening...")
        audio = recognizer.listen(source, timeout=5)
    try:
        query = recognizer.recognize_google(audio)
        return query
    except sr.UnknownValueError:
        print("Could not understand audio.")
        return None
    except sr.RequestError as e:
        print(
            "Could not request results from Google Speech Recognition service; {0}".format(
                e
            )
        )
        return None


@app.route("/")
def index():
    return render_template("index.html", conversation=conversation_history)


@app.route("/ask", methods=["POST"])
def ask():
    global fetched_data, prevQ, prevA  # Declare the variable to allow modification
    user_input = request.form["query"]

    # Add user input to conversation history
    conversation_history.append({"sender": "User", "message": user_input})

    # Check if the data has already been fetched
    if fetched_data is None:
        # Generate the topic and relevant information
        topic = identify_topic(user_input)
        fetched_data = fetch_product_info(topic)  # Fetch the data once

        if fetched_data is None:
            response_text = "No product information available."
            conversation_history.append({"sender": "AI", "message": response_text})
            return jsonify({"conversation": conversation_history})

    # Generate a passage using the fetched data
    passage = fetched_data
    prompt = make_prompt(user_input, passage, prevQ, prevA)
    print(prompt)
    answer = model.generate_content(prompt)
    response_text = answer._result.candidates[0].content.parts[0].text
    prevQ = user_input
    prevA = response_text

    # Add AI response to conversation history
    conversation_history.append({"sender": "AI", "message": response_text})

    # Convert the response text to speech and play it
    text_to_speech(response_text)

    # Return the updated conversation history to the front-end
    return jsonify({"conversation": conversation_history})


@app.route("/mic_click", methods=["POST"])
def mic_click():
    user_input = speech_to_text()
    if user_input:
        return jsonify({"input_text": user_input})
    return jsonify({"error": "Could not recognize speech."})


if __name__ == "__main__":
    app.run(debug=True)
