import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import pandas as pd

# Sample data - Vietnamese tech products
sample_sources = [
    {
        "Tên": "iPhone 15 Pro Max",
        "Mô tả": "Điện thoại thông minh cao cấp với chip A17 Pro, camera 48MP, màn hình Super Retina XDR, hỗ trợ 5G và Face ID"
    },
    {
        "Tên": "Samsung Galaxy S24 Ultra", 
        "Mô tả": "Smartphone Android flagship với bút S Pen, camera zoom 100x, màn hình Dynamic AMOLED 2X, vi xử lý Snapdragon 8 Gen 3"
    },
    {
        "Tên": "MacBook Air M3",
        "Mô tả": "Laptop siêu mỏng nhẹ với chip M3, màn hình Liquid Retina 13.6 inch, pin 18 giờ, thiết kế nhôm cao cấp"
    },
    {
        "Tên": "Dell XPS 13",
        "Mô tả": "Ultrabook Windows cao cấp với Intel Core i7, màn hình InfinityEdge 4K, thiết kế carbon fiber, cổng Thunderbolt 4"
    },
    {
        "Tên": "iPad Pro 12.9",
        "Mô tả": "Máy tính bảng chuyên nghiệp với chip M2, màn hình Liquid Retina XDR, hỗ trợ Apple Pencil và Magic Keyboard"
    },
    {
        "Tên": "Surface Pro 9",
        "Mô tả": "Tablet 2-in-1 Windows với Intel Core i5, màn hình cảm ứng 13 inch, bàn phím rời, bút Surface Pen tích hợp"
    },
    {
        "Tên": "Sony WH-1000XM5",
        "Mô tả": "Tai nghe chống ồn cao cấp với công nghệ AI, driver 30mm, pin 30 giờ, kết nối Bluetooth 5.2"
    },
    {
        "Tên": "AirPods Pro 2",
        "Mô tả": "Tai nghe không dây true wireless với chip H2, chống ồn chủ động, âm thanh không gian, case sạc MagSafe"
    },
    {
        "Tên": "Nintendo Switch OLED",
        "Mô tả": "Console game cầm tay với màn hình OLED 7 inch, dock TV, Joy-Con controllers, chơi được cả ở nhà và di động"
    },
    {
        "Tên": "PlayStation 5",
        "Mô tả": "Máy chơi game thế hệ mới với SSD siêu nhanh, ray tracing, DualSense controller haptic feedback, 4K gaming"
    },
    {
        "Tên": "Canon EOS R6 Mark II",
        "Mô tả": "Máy ảnh mirrorless full-frame với cảm biến 24.2MP, quay video 4K, chống rung IBIS, autofocus Dual Pixel"
    },
    {
        "Tên": "GoPro Hero 12",
        "Mô tả": "Action camera chống nước với video 5.3K, chống rung HyperSmooth, GPS tích hợp, điều khiển bằng giọng nói"
    }
]

# Initialize embedding model (you might need to install: pip install sentence-transformers)
print("Đang tải embedding model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # English model, but works okay with Vietnamese

def tfidf_search(question: str, sources: List[Dict], top_k: int = 3) -> List[tuple]:
    """TF-IDF search only"""
    texts = [f"{src['Tên']} {src['Mô tả']}" for src in sources]
    
    vectorizer = TfidfVectorizer().fit(texts + [question])
    tfidf_matrix = vectorizer.transform(texts)
    question_vec = vectorizer.transform([question])
    tfidf_sim = cosine_similarity(question_vec, tfidf_matrix).flatten()
    
    # Get top results with scores
    top_indices = tfidf_sim.argsort()[-top_k:][::-1]
    results = [(sources[i], tfidf_sim[i]) for i in top_indices]
    return results

def hybrid_search(question: str, sources: List[Dict], top_k: int = 3) -> List[tuple]:
    """Hybrid search: TF-IDF + Embedding"""
    texts = [f"{src['Tên']} {src['Mô tả']}" for src in sources]

    # 1. TF-IDF + cosine similarity
    vectorizer = TfidfVectorizer().fit(texts + [question])
    tfidf_matrix = vectorizer.transform(texts)
    question_vec = vectorizer.transform([question])
    tfidf_sim = cosine_similarity(question_vec, tfidf_matrix).flatten()

    # 2. Embedding vector search
    text_embeddings = embedding_model.encode(texts)
    question_embedding = embedding_model.encode([question])[0]
    emb_sim = np.dot(text_embeddings, question_embedding) / (
        np.linalg.norm(text_embeddings, axis=1) * np.linalg.norm(question_embedding) + 1e-8
    )

    # 3. Normalize scores
    scaler = MinMaxScaler()
    tfidf_sim_scaled = scaler.fit_transform(tfidf_sim.reshape(-1, 1)).flatten()
    emb_sim_scaled = scaler.fit_transform(emb_sim.reshape(-1, 1)).flatten()

    # 4. Combine scores
    hybrid_score = 0.5 * tfidf_sim_scaled + 0.5 * emb_sim_scaled

    # 5. Get top results with scores
    top_indices = hybrid_score.argsort()[-top_k:][::-1]
    results = [(sources[i], hybrid_score[i]) for i in top_indices]
    return results

def print_results(results: List[tuple], method_name: str):
    """Print search results in a nice format"""
    print(f"\n🔍 {method_name} Results:")
    print("-" * 50)
    for i, (source, score) in enumerate(results, 1):
        print(f"{i}. {source['Tên']} (Score: {score:.4f})")
        print(f"   📝 {source['Mô tả']}")
        print()

def compare_searches(question: str, sources: List[Dict], top_k: int = 3):
    """Compare TF-IDF and Hybrid search results"""
    print(f"❓ Câu hỏi: '{question}'")
    print("=" * 80)
    
    # TF-IDF search
    tfidf_results = tfidf_search(question, sources, top_k)
    print_results(tfidf_results, "TF-IDF Search")
    
    # Hybrid search
    hybrid_results = hybrid_search(question, sources, top_k)
    print_results(hybrid_results, "Hybrid Search")
    
    # Show difference
    tfidf_names = [result[0]['Tên'] for result in tfidf_results]
    hybrid_names = [result[0]['Tên'] for result in hybrid_results]
    
    print("📊 So sánh kết quả:")
    print(f"TF-IDF top 3: {tfidf_names}")
    print(f"Hybrid top 3: {hybrid_names}")
    
    different_results = set(tfidf_names) ^ set(hybrid_names)
    if different_results:
        print(f"🔄 Khác biệt: {different_results}")
    else:
        print("✅ Kết quả giống nhau")
    
    print("\n" + "="*80 + "\n")

# Test cases
test_questions = [
    "Tôi muốn mua điện thoại có camera tốt",  # Should favor phones with good cameras
    "Laptop nào phù hợp cho công việc văn phòng?",  # Should find laptops
    "Thiết bị gaming tốt nhất",  # Should find gaming devices
    "Tai nghe chống ồn cao cấp",  # Should find noise-canceling headphones
    "Máy ảnh chụp hình đẹp",  # Should find cameras
    "Tablet để vẽ và thiết kế",  # Should find tablets good for design
    "iPhone mới nhất",  # Exact keyword match
    "Sản phẩm Apple",  # Brand-based search
]

if __name__ == "__main__":
    print("🚀 Bắt đầu so sánh TF-IDF vs Hybrid Search")
    print("Loading embedding model... (có thể mất vài giây)")
    
    # Run comparison for each test question
    for question in test_questions:
        try:
            compare_searches(question, sample_sources)
        except Exception as e:
            print(f"❌ Lỗi với câu hỏi '{question}': {e}")
            continue
