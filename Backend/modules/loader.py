import pickle

def load_data():
    with open("data/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    with open("data/metadatas.pkl", "rb") as f:
        metadatas = pickle.load(f)

    with open("data/bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    return texts, metadatas, bm25