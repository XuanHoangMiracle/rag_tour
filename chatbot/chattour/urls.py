# chattour/urls.py
from django.urls import path
from .views import chat_view, clear_chat_history, get_chat_history 

urlpatterns = [
    path('', chat_view, name='chat-view'),
    path('api/chat/clear', clear_chat_history, name='clear_chat'),
    path('api/chat/history', get_chat_history, name='get_history'), 
]
