# novel_qa.py
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import torch

# ========== モデル設定 ==========
LLM_NAME = "cyberagent/open-calm-small"
EMBED_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ========== 1. LLM & トークナイザー ==========
print("Loading LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    LLM_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ========== 2. 埋め込みモデル ==========
print("Loading embedder...")
embedder = SentenceTransformer(EMBED_NAME)

# ========== 3. テキスト分割 ==========
with open("wagahaiwa_nekodearu.txt", encoding="utf-8") as f:
    text = f.read()

# 適度に分割（1チャンク = 約300文字）
chunk_size = 300
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
print(f"Total chunks: {len(chunks)}")

# ========== 4. ベクトル化 & FAISS ==========
embeddings = embedder.encode(chunks, show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"FAISS index size: {index.ntotal}")

# ========== 5. 質問応答関数 ==========
def ask(question, top_k=3):
    q_emb = embedder.encode([question])
    D, I = index.search(q_emb, top_k)
    context = "\n".join(chunks[i] for i in I[0])

    prompt = f"""以下の小説を読んで、質問に簡潔に短く日本語で答えてください。

{context}

質問: {question}
回答:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items() if k != "token_type_ids"}

    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,      # サンプル生成
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        repetition_penalty=1.5

    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=====================================")
    print(answer)
    print("=====================================")


# ========== 6. 実行例 ==========
ask("吾輩とは誰かのことか？")
ask("吾輩はどこに向かった？")

