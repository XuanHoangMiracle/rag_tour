from django.shortcuts import render

import pandas as pd
from sentence_transformers import SentenceTransformer
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rest_framework.decorators import api_view
from rest_framework import viewsets
from rest_framework.response import Response

import ollama


# Đọc stop words
with open(f'F:\\RAG_DA\\Rag_Travel_Tour\\stop_words_Vietnamese.txt', encoding='utf-8') as f:
    stop_words = set([line.strip() for line in f if line.strip()])

# Hàm loại bỏ stop words
def remove_stop_words(text):
    words = re.findall(r'\w+', text.lower())
    filtered = [w for w in words if w not in stop_words]
    return ' '.join(filtered)

# Đọc dữ liệu tour
df = pd.read_csv('F:\\RAG_DA\\Rag_Travel_Tour\\tour.csv')

# Chọn các trường cần embeding name,location,time,cost,services
fields = ['name', 'location', 'time', 'cost', 'services']
df['text'] = df[fields].astype(str).agg(' '.join, axis=1)
df['text_clean'] = df['text'].apply(remove_stop_words)

# Embeding
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = model.encode(df['text_clean'].tolist())

# Lưu embedding nếu cần
np.save('tour_embeddings.npy', embeddings)

def embed_query(query, model, stop_words):
    query_clean = remove_stop_words(query)
    query_embedding = model.encode([query_clean])[0]
    return query_embedding

def search_tours(query_embedding, tour_embeddings, df, top_k=3):
    similarities = cosine_similarity([query_embedding], tour_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return df.iloc[top_indices], similarities[top_indices]

def generate_answer(user_query, retrieved_tours):
    # Tạo prompt cho mô hình
    context = "\n".join([
        f"Tour: {row['name']}\nĐịa điểm: {row['location']}\nThời gian: {row['time']}\nGiá: {row['cost']}\nDịch vụ: {row['services']}"
        for _, row in retrieved_tours.iterrows()
    ])
    prompt = f"""Bạn là trợ lý tư vấn tour du lịch và Luôn trả lời bằng tiếng việt. Chỉ sử dụng thông tin dưới đây để trả lời câu hỏi của khách hàng. 
Nếu không tìm thấy thông tin phù hợp, hãy trả lời: 'Xin lỗi, tôi không tìm thấy thông tin phù hợp trong dữ liệu tour.'
Thông tin tour:
{context}

Câu hỏi khách hàng: {user_query}
"""
    response = ollama.chat(model='llama3', messages=[
        {'role': 'user', 'content': prompt}
    ])
    return response['message']['content']

tour_embeddings = np.load('tour_embeddings.npy')



@api_view(['GET'])
def chat_view(request):
    user_query = request.query_params.get('query', '')
    if not user_query:
        return Response({"error": "Query parameter is required."}, status=400)

    query_embedding = embed_query(user_query, model, stop_words)
    retrieved_tours, scores = search_tours(query_embedding, tour_embeddings, df, top_k=5)
    answer = generate_answer(user_query, retrieved_tours)

    return Response({
        "query": user_query,
        "answer": answer,
        "retrieved_tours": retrieved_tours.to_dict(orient='records'),
        "scores": scores.tolist()
    })