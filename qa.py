# novel_qa_ollama_interactive_bge_cache_v3.py
from sentence_transformers import SentenceTransformer
import faiss
import subprocess
import numpy as np
import os

# ========== 設定 ==========
OLLAMA_MODEL = "schroneko/gemma-2-2b-jpn-it:latest"
EMBED_NAME = "BAAI/bge-m3"
CHUNK_SIZE = 300
INDEX_FILE = "novel_index.faiss"
CHUNKS_FILE = "novel_chunks.npy"
EMB_FILE = "novel_emb.npy"
TEXT_FILE = "wagahaiwa_nekodearu.txt"

# ========== 埋め込みモデル ==========
print("Loading embedder...")
embedder = SentenceTransformer(EMBED_NAME)

# ========== テキスト分割 ==========
with open(TEXT_FILE, encoding="utf-8") as f:
    text = f.read()

chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
print(f"Total chunks: {len(chunks)}")

# ========== FAISS & 埋め込みロード or 作成 ==========
if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
    print("Loading cached FAISS index and chunks...")
    index = faiss.read_index(INDEX_FILE)
    chunks_cached = np.load(CHUNKS_FILE, allow_pickle=True)
    
    # 安全チェック
    if len(chunks_cached) != index.ntotal:
        print("⚠️ キャッシュのチャンク数とインデックスベクトル数が一致しません。再作成します。")
        os.remove(INDEX_FILE)
        os.remove(CHUNKS_FILE)
        if os.path.exists(EMB_FILE):
            os.remove(EMB_FILE)
        cached_exists = False
    else:
        chunks = chunks_cached
        cached_exists = True
else:
    cached_exists = False

if not cached_exists:
    print("Encoding chunks (first time or cache mismatch, please wait)...")
    embeddings = embedder.encode(
        chunks,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    # 再チェック
    assert index.ntotal == len(chunks), "FAISSインデックスのサイズとチャンク数が一致しません！"
    
    faiss.write_index(index, INDEX_FILE)
    np.save(CHUNKS_FILE, chunks)
    np.save(EMB_FILE, embeddings)
    print(f"Saved FAISS index and embeddings. Index ntotal: {index.ntotal}")

print(f"FAISS index size: {index.ntotal}")

# ========== Ollama QA 関数 ==========
def ask(question, top_k=3, max_context_chars=1000):
    q_emb = embedder.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    context = "\n".join(chunks[i] for i in I[0])

    if len(context) > max_context_chars:
        context = context[-max_context_chars:]

    prompt = f"""以下の小説を読んで、質問に簡潔に短く日本語で答えてください。

{context}

質問: {question}
回答:"""

    result = subprocess.run(
        ["ollama", "run", OLLAMA_MODEL],
        input=prompt,
        capture_output=True,
        text=True
    )

    # 本当にエラーのときだけ表示
    if result.stderr and "Error" in result.stderr:
        print("=== Ollama STDERR ===")
        print(result.stderr)

    answer = result.stdout.strip()
    answer = answer.replace("</start_of_turn>", "").strip()
    return answer

# ========== インタラクティブ質問ループ ==========
print("\n小説QAインタラクティブモード開始（終了するには 'exit' と入力）")
while True:
    question = input("\n質問を入力してください: ").strip()
    if question.lower() == "exit":
        print("終了します。")
        break
    if question == "":
        print("質問が空です。もう一度入力してください。")
        continue

    answer = ask(question)
    print(f"\n回答: {answer}")

