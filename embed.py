from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

text = ["Roshan Roger Roby Robin"]
embeddings = model.encode(text)
print(embeddings)
