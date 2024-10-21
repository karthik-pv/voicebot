import google.generativeai as genai
import chromadb
import json
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
from chromadb import Documents, EmbeddingFunction, Embeddings
from flask import Flask, render_template, request, jsonify
from utils import fetch_product_info
import time

app = Flask(__name__)


class GeminiEmbeddingFunction(EmbeddingFunction):
    def _call_(self, input: Documents) -> Embeddings:
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(
            model=model, content=input, task_type="retrieval_document", title=title
        )["embedding"]


model = genai.GenerativeModel("gemini-pro")
counter = 0
db = None
passage = ""
prevQ = ""
prevA = ""


def create_chroma_db(documents, name):
    print(documents)
    chroma_client = chromadb.Client()
    db = chroma_client.create_collection(
        name=name, embedding_function=GeminiEmbeddingFunction()
    )
    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))
    return db


genai.configure(api_key="AIzaSyCadPuPUQvtH-NsETbzmgooO9OT2NkAt1s")


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        user_input = request.form.get("input_text")
        if user_input:
            response = process_input(user_input)
            return jsonify(response)

    return render_template("index.html")


@app.route("/mic_click", methods=["POST"])
def mic_click():
    user_input = speech_to_text()
    if user_input:
        response = process_input(user_input)
        return jsonify(response)
    return jsonify({"error": "Could not recognize speech."})


def process_input(user_input):
    global counter, db, passage, prevQ, prevA  # Declare prevQ and prevA as global
    if user_input.lower() == "exit":
        return {"message": "Exiting."}

    if counter == 0:
        topic = identify_topic(user_input, model)
        data = fetch_product_info(topic)
        if data is None:
            return {"response": "No product information available."}
        db = create_chroma_db(data, "content")
        passage = get_relevant_passage(user_input, db)
        # passage = data
        counter += 1

    prompt = make_prompt(
        user_input, passage, prevQ, prevA
    )  # Pass prevQ and prevA to make_prompt
    answer = model.generate_content(prompt)
    response_text = answer._result.candidates[0].content.parts[0].text
    text_to_speech(response_text)

    # Update prevQ and prevA after generating the response
    prevQ = user_input
    prevA = response_text

    return {"response": response_text}


def get_relevant_passage(query, db):
    results = db.query(query_texts=[query], n_results=7)
    passages = results["documents"][0]
    return passages


def make_prompt(query, relevant_passage, prevQ, prevA):
    prompt = (
        """You are a helpful and informative bot that answers questions using text from the reference passage included below. \
        Be sure to respond in a complete sentence, being comprehensive, don't include all relevant background information just the bare minimum. \
        However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
        strike a friendly and conversational tone. \
        If the passage is irrelevant to the answer, you may ignore it.\
        If the data is not sufficient just return the response text as 'Data Insufficient'....nothing else\
        The response must be short....around 75-100 words \
        When you are not talking about the product in particular keep your responses short \
        Look at the previous question and answer and give the answer in the right context with respect to the previous interaction \
        Answer the question as if you are a human salesman......end every response with a question to help the user further.....example volunteer extra information.....volunteer to compare the products and provide the best
        QUESTION: '{query}'
        PASSAGE: '{relevant_passage}'
        'The previous question was - {prevQ}\
        'The previous answer was - {prevA}'

        ANSWER:
        """
    ).format(query=query, relevant_passage=relevant_passage, prevQ=prevQ, prevA=prevA)
    print(prompt)
    return prompt


def identify_topic(query, model):
    prompt = (
        "from the given sentence identify the topic of discussion.......the topic will mostly always be a product......give me the \
        range of products the user is looking for eg if search query is how good is samsung m32.....you need to return samsung m32......in max of \
        3 words just tell me the topic.....i want your response to be as accurate as possible \
        sentence : '{query}'"
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
        print(f"You said: {query}")
        return query
    except sr.UnknownValueError:
        print("Sorry, I could not understand the audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return None


if __name__ == "__main__":
    app.run(debug=True)
