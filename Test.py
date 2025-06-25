import json
import requests
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Dummy data mẫu (bạn thay bằng dữ liệu thực tế)
sources = [
    {
        "STT": 1,
        "ID": "001",
        "Tên": "Nguồn A",
        "Link-API": "http://api-a.com",
        "JSON": {},
        "Mô tả": "Nguồn A cung cấp dữ liệu về thời tiết.",
        "Quota": 1000,
        "Ranking": 1
    },
    {
        "STT": 2,
        "ID": "002",
        "Tên": "Nguồn B",
        "Link-API": "http://api-b.com",
        "JSON": {},
        "Mô tả": "Nguồn B chuyên về dữ liệu tài chính, báo cáo chi phí, doanh thu.",
        "Quota": 500,
        "Ranking": 2
    },
    {
        "STT": 3,
        "ID": "003",
        "Tên": "Nguồn C",
        "Link-API": "http://api-c.com",
        "JSON": {},
        "Mô tả": "Nguồn C cung cấp dữ liệu về nhân sự, thông tin nhân viên.",
        "Quota": 300,
        "Ranking": 3
    },
    {
        "STT": 4,
        "ID": "004",
        "Tên": "Nguồn D",
        "Link-API": "http://api-d.com",
        "JSON": {},
        "Mô tả": "Nguồn D chuyên về dữ liệu khách hàng, thông tin liên hệ.",
        "Quota": 200,
        "Ranking": 4
    },
    # ... thêm nguồn khác ...
]

def hybrid_search(question: str, sources: List[Dict], top_k: int = 2) -> List[Dict]:
    # Sử dụng TF-IDF + cosine similarity trên trường 'Tên' và 'Mô tả'
    texts = [f"{src['Tên']} {src['Mô tả']}" for src in sources]
    vectorizer = TfidfVectorizer().fit(texts + [question])
    tfidf_matrix = vectorizer.transform(texts)
    question_vec = vectorizer.transform([question])
    similarities = cosine_similarity(question_vec, tfidf_matrix).flatten()
    # Lấy top_k nguồn phù hợp nhất
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [sources[i] for i in top_indices]

def build_prompt(question: str, matched_sources: List[Dict]) -> str:
    context = "\n".join([f"{s['Tên']}: {s['Mô tả']}" for s in matched_sources])
    prompt = (
        f"Dưới đây là các nguồn dữ liệu:\n{context}\n\n"
        f"Câu hỏi: {question}\n"
        f"Hãy trả lời ngắn gọn tên nguồn dữ liệu phù hợp nhất với câu hỏi trên. "
        f"Chỉ trả về duy nhất tên nguồn, không giải thích thêm."
    )
    return prompt

def query_llm(prompt: str, api_url: str) -> str:
    headers = {"Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_tokens": 100, "temperature": 0.5}
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        response.raise_for_status()
        result = response.json()
        print(result)  # Thêm dòng này để kiểm tra cấu trúc trả về
        # Sửa phần lấy text cho phù hợp với response thực tế
        if "text" in result:
            return result["text"]
        elif "choices" in result and isinstance(result["choices"], list):
            return result["choices"][0].get("text", "Không nhận được phản hồi từ LLM.")
        else:
            return str(result)
    except Exception as e:
        return f"Lỗi khi gọi LLM: {e}"

def main(question: str, api_url: str):
    matched_sources = hybrid_search(question, sources)
    prompt = build_prompt(question, matched_sources)
    answer = query_llm(prompt, api_url)
    print("Câu trả lời từ LLM:")
    print(answer)

if __name__ == "__main__":
    # Ví dụ sử dụng
    user_question = "Nguồn chuyên về dữ liệu thời tiết"
    lmstudio_api_url = "http://192.168.102.16:1234/v1/completions"  # Thay bằng URL thực tế của LMstudio
    main(user_question, lmstudio_api_url)