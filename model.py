import google.generativeai as genai

import chromadb
import numpy as np
import pandas as pd

from utils import fetch_product_info

from IPython.display import Markdown
from chromadb import Documents, EmbeddingFunction, Embeddings


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

data = [
    "M-Audio Oxygen Pro Mini 32 Key USB MIDI Keyboard Controller With Beat Pads, MIDI assignable Knobs, Buttons & Faders and Software Suite Included  4.4 out of 5 stars 12,489",
    "M-Audio Oxygen 61 V 61 Key USB MIDI Keyboard Controller With Beat Pads, Smart Chord & Scale Modes, Arpeggiator and Software Suite Included  4.4 out of 5 stars 20,300",
    "M-Audio Oxygen 49 V 49 Key USB MIDI Keyboard Controller With Beat Pads, Smart Chord & Scale Modes, Arpeggiator and Software Suite Included  4.4 out of 5 stars 15,899",
    "M-Audio Oxygen Pro 49 49 Key USB MIDI Keyboard Controller With Beat Pads, MIDI assignable Knobs, Buttons & Faders and Software Suite Included  4.6 out of 5 stars 21,900",
    "Vault Ikon MK2 49 Key Velocity Sensitive Midi Keyboard with Bitwig 8-Track Software  3.7 out of 5 stars 7,123",
    "M-Audio Keystation 88 MK3 88 Key Semi Weighted MIDI Keyboard Controller for Complete Command of Virtual Synthesisers and DAW parameters  4.5 out of 5 stars 28,000",
]

db = create_chroma_db(data, "midicontrollers")

peek_data = db.peek(3)


def get_relevant_passage(query, db):
    results = db.query(query_texts=[query], n_results=5)
    passages = results["documents"][0]
    return passages


def make_prompt(query, relevant_passage):
    prompt = (
        """You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, dont include all relevant background information just the bare minimum. \
  However, you are talking to a non-technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. \
  If the passage is irrelevant to the answer, you may ignore it.\
  The response must be short....around 150-200 words \
  Answer the question as if you are a human salesman......end every response with a question to help the user further.....example volunteer extra information.....volunteer to compare the products and provide the best
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """
    ).format(query=query, relevant_passage=relevant_passage)

    return prompt


passage = get_relevant_passage("midi controller", db)

query = "yes do compare the 2"
prompt = make_prompt(query, passage)
Markdown(prompt)
print(prompt)


model = genai.GenerativeModel("gemini-pro")
answer = model.generate_content(prompt)
print(answer)
