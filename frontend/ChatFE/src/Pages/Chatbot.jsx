// React Chatbot UI for Django RAG
// ------------------------------------------------------------
// How to use
// 1) Put this file in your React app (e.g., src/Chatbot.jsx)
// 2) Ensure Tailwind (or your own CSS) is available for basic styling.
// 3) Create .env file in your React project root and set:
//      VITE_CHAT_API_URL=http://localhost:8000/chat/   // or your actual endpoint
//    The component will POST { question: "..." } to this URL.
// 4) In your Django app, expose an endpoint (examples below) that returns JSON:
//      { "answer": "..." }
// 5) Import and render <Chatbot /> anywhere in your app.
// ------------------------------------------------------------

import React, { useEffect, useMemo, useRef, useState } from "react";

const API_URL = import.meta.env.VITE_CHAT_API_URL || "http://localhost:8000/chat/";

/**
 * Minimal expected Django view (example):
 * 
 * from django.views.decorators.http import require_POST
 * from django.http import JsonResponse
 * import json
 * 
 * @require_POST
 * def chat_view(request):
 *     try:
 *         data = json.loads(request.body.decode('utf-8'))
 *         question = data.get('question', '').strip()
 *         if not question:
 *             return JsonResponse({"error": "Empty question"}, status=400)
 * 
 *         # TODO: Call your RAG + Llama3 logic here, get 'answer'
 *         answer = rag_answer(question)  # <- your function
 * 
 *         return JsonResponse({"answer": answer})
 *     except Exception as e:
 *         return JsonResponse({"error": str(e)}, status=500)
 * 
 * # urls.py
 * from django.urls import path
 * from .views import chat_view
 * urlpatterns = [ path('', chat_view, name='chat'), ]
 * 
 * # If CORS is needed (frontend on different port):
 * #   pip install django-cors-headers
 * #   settings.py: add 'corsheaders' to INSTALLED_APPS and middleware, 
 * #   CORS_ALLOW_ALL_ORIGINS = True  (or set specific origins)
 */

export default function Chatbot() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Xin chào! Mình là trợ lý du lịch. Hãy hỏi về tour, giá, lịch trình…" },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const scrollerRef = useRef(null);

  useEffect(() => {
    // Auto-scroll to latest message
    if (scrollerRef.current) {
      scrollerRef.current.scrollTop = scrollerRef.current.scrollHeight;
    }
  }, [messages, loading]);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  async function sendMessage(e) {
    e?.preventDefault();
    setError("");
    const text = input.trim();
    if (!text) return;

    // Push user message to UI
    setMessages(prev => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text }),
      });

      let data = null;
      try {
        data = await res.json();
      } catch (_) {
        // If server returns non-JSON, handle gracefully
        throw new Error(`Server returned status ${res.status}`);
      }

      if (!res.ok) {
        const msg = data?.error || `Request failed with status ${res.status}`;
        throw new Error(msg);
      }

      const answer = data?.answer || data?.message || "(Không có nội dung trả lời)";
      setMessages(prev => [...prev, { role: "assistant", content: answer }]);
    } catch (err) {
      setError(err.message || String(err));
      setMessages(prev => [
        ...prev,
        { role: "assistant", content: "Xin lỗi, đã có lỗi xảy ra khi xử lý yêu cầu của bạn." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey && canSend) {
      sendMessage(e);
    }
  }

  return (
    <div className="min-h-screen w-full bg-gray-50 flex items-center justify-center p-4">
      <div className="w-full max-w-3xl rounded-2xl shadow-lg bg-white border border-gray-200 flex flex-col h-[90vh]">
        <Header />

        <div ref={scrollerRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {messages.map((m, idx) => (
            <Bubble key={idx} role={m.role} content={m.content} />
          ))}
          {loading && <TypingBubble />}
        </div>

        {error && (
          <div className="px-4 pb-2 text-sm text-red-600">Lỗi: {error}</div>
        )}

        <form onSubmit={sendMessage} className="p-4 border-t border-gray-200">
          <div className="flex items-end gap-2">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              rows={1}
              placeholder="Nhập câu hỏi của bạn..."
              className="flex-1 resize-none rounded-xl border border-gray-300 p-3 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              type="submit"
              disabled={!canSend}
              className="rounded-xl px-4 py-3 bg-blue-600 text-white disabled:opacity-50 hover:bg-blue-700 transition"
            >
              Gửi
            </button>
          </div>

          <Suggested onPick={(q) => setInput(q)} />

          <div className="mt-2 text-[11px] text-gray-500">
            API: <code>{API_URL}</code>
          </div>
        </form>
      </div>
    </div>
  );
}

function Header() {
  return (
    <div className="p-4 border-b border-gray-200 flex items-center gap-3">
      <Logo />
      <div className="flex flex-col">
        <div className="font-semibold">TravelTour Chatbot</div>
        <div className="text-xs text-gray-500">RAG + Llama3 (Ollama) qua Django API</div>
      </div>
    </div>
  );
}

function Logo() {
  return (
    <div className="h-10 w-10 rounded-xl bg-blue-600/10 flex items-center justify-center">
      <span className="text-blue-700 font-bold">TT</span>
    </div>
  );
}

function Bubble({ role, content }) {
  const isUser = role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`${
          isUser ? "bg-blue-600 text-white" : "bg-gray-100 text-gray-900"
        } max-w-[80%] whitespace-pre-wrap px-3 py-2 rounded-2xl ${
          isUser ? "rounded-br-sm" : "rounded-bl-sm"
        }`}
      >
        {content}
      </div>
    </div>
  );
}

function TypingBubble() {
  return (
    <div className="flex justify-start">
      <div className="bg-gray-100 text-gray-900 px-3 py-2 rounded-2xl rounded-bl-sm">
        <div className="flex items-center gap-1">
          <Dot /> <Dot delay={150} /> <Dot delay={300} />
        </div>
      </div>
    </div>
  );
}

function Dot({ delay = 0 }) {
  return (
    <span
      className="inline-block w-2 h-2 rounded-full bg-gray-500 animate-pulse"
      style={{ animationDelay: `${delay}ms` }}
    />
  );
}

function Suggested({ onPick }) {
  const suggestions = [
    "Tìm tour rẻ nhất đi Đà Nẵng",
    "Gợi ý tour 3 ngày 2 đêm dưới 5 triệu",
    "Dịch vụ bao gồm của tour Pleiku?",
  ];

  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {suggestions.map((s, i) => (
        <button
          key={i}
          type="button"
          onClick={() => onPick(s)}
          className="text-xs px-3 py-1.5 rounded-full border border-gray-300 hover:bg-gray-50"
        >
          {s}
        </button>
      ))}
    </div>
  );
}
