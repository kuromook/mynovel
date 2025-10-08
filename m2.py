import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss

# ----------------------------------------
# 1. 小説テキストを読み込む
# ----------------------------------------
novel_folder = Path("./novels")  # 小説テキストを置くフォルダ
chunks = []

for txt_file in novel_folder.glob("*.txt"):
    with open(txt_file, encoding="utf-8") as f:
        text = f.read()
        # 500文字ごとに分割
        step = 500
        chunks += [text[i:i+step] for i in range(0, len(text), step)]

print(f"Total chunks: {len(chunks)}")

# ----------------------------------------
# 2. 日本語対応埋め込みモデル
# ----------------------------------------
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", device="cuda")

# 3. テキストのベクトル化
embeddings = embed_model.encode(chunks, convert_to_tensor=True, show_progress_bar=True)
embeddings = embeddings.cpu().detach().numpy().astype("float32")  # FAISS は float32 必須

# ----------------------------------------
# 4. FAISS インデックス作成（GPU）
# ----------------------------------------
d = embeddings.shape[1]  # 埋め込み次元
res = faiss.StandardGpuResources()
index = faiss.IndexFlatL2(d)
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)

gpu_index.add(embeddings)
print(f"FAISS index size: {gpu_index.ntotal}")

# ----------------------------------------
# 5. Transformers で質問応答モデル
# ----------------------------------------
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(device)

# ----------------------------------------
# 6. QA 関数
# ----------------------------------------
def qa(query, top_k=3):
    # 1) ベクトル検索
    q_vec = embed_model.encode([query], convert_to_tensor=True).cpu().detach().numpy().astype("float32")
    D, I = gpu_index.search(q_vec, top_k)
    
    # 2) 上位のテキストを結合
    context = "\n".join([chunks[i] for i in I[0]])
    
    # 3) 質問 + コンテキストをモデルに入力
    input_text = f"Context: {context}\nQuestion: {query}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# ----------------------------------------
# 7. 質問テスト
# ----------------------------------------
print("日本語小説 QA システム。終了するには exit または quit と入力してください。")
while True:
    q = input("\n質問: ")
    if q.lower() in ["exit", "quit"]:
        break
    print("回答:", qa(q))

