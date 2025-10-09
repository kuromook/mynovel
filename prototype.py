# novel_qa_ollama_stdin.py
from sentence_transformers import SentenceTransformer
import faiss
import subprocess

# ========== 設定 ==========
OLLAMA_MODEL = "schroneko/gemma-2-2b-jpn-it:latest"
EMBED_NAME = "sonoisa/sentence-bert-base-ja-mean-tokens"

# ========== 埋め込みモデル ==========
print("Loading embedder...")
embedder = SentenceTransformer(EMBED_NAME)

# ========== テキスト分割 ==========
with open("wagahaiwa_nekodearu.txt", encoding="utf-8") as f:
    text = f.read()

chunk_size = 300
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
print(f"Total chunks: {len(chunks)}")

# ========== ベクトル化 & FAISS ==========
print("Encoding chunks...")
embeddings = embedder.encode(chunks, show_progress_bar=True, convert_to_numpy=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"FAISS index size: {index.ntotal}")

# ========== Ollama QA ==========
def ask(question, top_k=3, max_context_chars=1000):
    # 類似チャンク検索
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = index.search(q_emb, top_k)
    context = "\n".join(chunks[i] for i in I[0])

    if len(context) > max_context_chars:
        context = context[-max_context_chars:]

    prompt = f"""以下の小説を読んで、質問に簡潔に短く日本語で答えてください。

{context}

質問: {question}
回答:"""

    # stdin 経由で Ollama に渡す
    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        capture_output=True,
        text=True
    )

    if result.stderr:
        print("=== Ollama STDERR ===")
        print(result.stderr)

    answer = result.stdout.strip()
    print("=====================================")
    print(answer)
    print("=====================================")

# ========== 実行例 ==========
ask("吾輩とは誰かのことか？")
ask("吾輩はどこに向かった？")

