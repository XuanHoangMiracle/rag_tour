import json
import time
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import sys
import django

# --- CẤU HÌNH ĐƯỜNG DẪN (PATH) ---
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Thêm thư mục này vào sys.path để Python nhìn thấy folder 'chattour' bên trong
sys.path.append(current_dir)

# 3. Setup Django (Để chạy được các lệnh liên quan đến Database)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chatbot.settings")
django.setup()

# --- IMPORT SERVICE (SỬA ĐỔI QUAN TRỌNG) ---
print("🔌 Đang kết nối tới RAG Service...")
try:
    # Đường dẫn bạn cung cấp: Rag_Travel_Tour\chatbot\chattour\services.py
    # Vì ta đang đứng ở 'chatbot', nên gọi vào 'chattour.services'
    from chattour.services import rag_service
    print("✅ Đã import thành công: rag_service")
except ImportError as e:
    print(f"❌ Lỗi Import nghiêm trọng: {e}")
    print("💡 Gợi ý kiểm tra:")
    print("   1. File 'Rag_Travel_Tour/chatbot/chattour/services.py' có tồn tại không?")
    print("   2. Trong file đó, dòng cuối cùng có lệnh 'rag_service = RAGService()' không?")
    sys.exit(1)

# --- PHẦN DƯỚI GIỮ NGUYÊN ---
# (Từ đoạn tải NLTK và hàm calculate_metrics trở đi không cần sửa)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("⏳ Đang tải dữ liệu NLTK...")
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ... (Copy tiếp phần còn lại của code cũ vào đây)
print("⏳ Đang tải model đánh giá (SentenceTransformer)...")
similarity_model = SentenceTransformer('keepitreal/vietnamese-sbert') 
rouge_evaluator = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
chencherry = SmoothingFunction() 

def calculate_metrics(prediction, reference):
    # a. BLEU Score
    ref_tokens = [reference.split()]
    pred_tokens = prediction.split()
    bleu = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=chencherry.method1)
    
    # b. ROUGE-L
    rouge_score = rouge_evaluator.score(reference, prediction)['rougeL'].fmeasure
    
    # c. Semantic Similarity (Cosine)
    embeddings = similarity_model.encode([prediction, reference])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    return bleu, rouge_score, similarity

def run_evaluation(dataset_path='test_dataset.json'):
    base_path = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_path, dataset_path)
    
    print(f"📂 Đang đọc dữ liệu từ: {full_path}")
    
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
    except FileNotFoundError:
        print("❌ Lỗi: Không tìm thấy file test_dataset.json")
        return

    results = []
    print(f"🚀 Bắt đầu đánh giá trên {len(test_data)} mẫu...")
    
    for item in tqdm(test_data, desc="Processing"):
        start_time = time.time()
        try:
            response = rag_service.chat(item['question'], session_id=f"eval_auto_{item['id']}")
            chatbot_answer = response['answer']
        except Exception as e:
            print(f"\n❌ Lỗi mẫu ID {item['id']}: {e}")
            chatbot_answer = "Error"
            
        latency = time.time() - start_time
        bleu, rouge, similarity = calculate_metrics(chatbot_answer, item['ground_truth'])
        
        results.append({
            "id": item['id'],
            "type": item.get('type', 'general'),
            "question": item['question'],
            "ground_truth": item['ground_truth'],
            "chatbot_answer": chatbot_answer,
            "bleu": round(bleu, 4),
            "rouge_l": round(rouge, 4),
            "similarity": round(similarity, 4),
            "latency": round(latency, 2)
        })
        time.sleep(3) 

    output_file = 'evaluation_results.csv'
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ Đã xuất kết quả ra file: {output_file}")
    print("-" * 50)
    print("📊 KẾT QUẢ TRUNG BÌNH:")
    print(f"   ➤ BLEU Score: {df['bleu'].mean():.4f}")
    print(f"   ➤ ROUGE-L:    {df['rouge_l'].mean():.4f}")
    print(f"   ➤ Similarity: {df['similarity'].mean():.4f}")
    print(f"   ➤ Latency:    {df['latency'].mean():.2f}s")
    print("-" * 50)

if __name__ == "__main__":
    run_evaluation()