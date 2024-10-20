import google.generativeai as genai
import chromadb
import json
import speech_recognition as sr
from gtts import gTTS
import os
import playsound
from chromadb import Documents, EmbeddingFunction, Embeddings
import pyaudio
from utils import fetch_product_info


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __call__(self, input: Documents) -> Embeddings:
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(
            model=model, content=input, task_type="retrieval_document", title=title
        )["embedding"]


def create_chroma_db(documents, name):
    chroma_client = chromadb.Client()
    db = chroma_client.create_collection(
        name=name, embedding_function=GeminiEmbeddingFunction()
    )
    for i, d in enumerate(documents):
        db.add(documents=d, ids=str(i))
    return db


GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)


def get_relevant_passage(query, db):
    results = db.query(query_texts=[query], n_results=7)
    passages = results["documents"][0]
    return passages


def make_prompt(query, relevant_passage):
    prompt = (
        """You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, dont include all relevant background information just the bare minimum. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.\
  If the data is not sufficient just return the response text as 'Data Insufficient'....nothing else\
  The response must be short....around 75-100 words \
  When you are not talking about the product in particular keep your responses short \
  Answer the question as if you are a human salesman......end every response with a question to help the user further.....example volunteer extra information.....volunteer to compare the products and provide the best
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """
    ).format(query=query, relevant_passage=relevant_passage)

    return prompt


def identify_topic(query, model):
    prompt = (
        "from the given sentence identify the topic of discussion.......the topic will mostly always be a product......give me the \
    range of products the user is looking for eg if search query is how good is samsung m32.....you need to return samsung m32......in max of \
    3 words just tell me the topic.....i want your response to be as accurate as possible \
    sentence : '{query}'"
    ).format(query=query)

    # Generate content
    topic = model.generate_content(prompt)
    print(topic)
    # Print for debugging to inspect the structure
    # print("DEBUG: Topic content structure: ", topic)

    # Try extracting the text safely
    try:
        text = topic._result.candidates[0].content.parts[0].text
        return text
    except AttributeError:
        print(
            "AttributeError: The topic response did not contain the expected structure."
        )
        return None


def text_to_speech(text):
    tts = gTTS(text=text, lang="en-us", tld="us", slow=False)
    filename = "response.mp3"
    tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)


def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone(
        sample_rate=16000, chunk_size=1024
    ) as source:  # Adjusting sample rate and chunk size
        print("Listening...")
        audio = recognizer.listen(
            source, timeout=5
        )  # Set a timeout for faster response
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


counter = 0
model = genai.GenerativeModel("gemini-pro")
passage = ""
db = None

while True:
    user_input = speech_to_text()
    if user_input is None:
        continue
    if user_input.lower() == "exit":
        break
    if counter == 0:
        topic = identify_topic(user_input, model)
        print(f"Identified Topic: {topic}")
        data = fetch_product_info(topic)
        if data is None:
            text_to_speech("No product information available.")
            continue
        db = create_chroma_db(data, "content")
        passage = get_relevant_passage(user_input, db)
        counter += 1
    prompt = make_prompt(user_input, passage)
    answer = model.generate_content(prompt)
    response_text = answer._result.candidates[0].content.parts[0].text
    print(response_text)
    text_to_speech(response_text)

# while True:
#     user_input = input("You: ")
#     if user_input == "exit":
#         break
#     if counter == 0:
#         topic = identify_topic(user_input, model)
#         print(topic)
#         data = fetch_product_info(topic)
#         db = create_chroma_db(data, "content")
#         passage = get_relevant_passage(user_input, db)
#         counter += 1
#     prompt = make_prompt(user_input, passage)
#     answer = model.generate_content(prompt)
#     print(answer._result.candidates[0].content.parts[0].text)
