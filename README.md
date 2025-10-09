### 小説のテキストファイルを読み込み、Q&Aで返せるようにするAI  

python3.10 / CUDA11.8 /GTX1060 / ubuntu 22  
ollamaのモデルは ollama CLI で pull   
今後はPCのスペックアップで検討 
  
prototype.py : Q&Aスクリプトの原型・最小構成  
qa.py : Q&Aをループ処理・キャッシュの追加による高速化等の機能強化
