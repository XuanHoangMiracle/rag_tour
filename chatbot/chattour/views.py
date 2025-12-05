# views/chat_views.py
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .services import rag_service
import uuid


@api_view(['GET', 'POST'])
def chat_view(request):
    """
    API endpoint để chat với Gemini (có chat history)
    
    GET: /api/chat?query=Tour Nha Trang&session_id=user123
    POST: /api/chat
    Body: {
        "query": "Tour Nha Trang giá rẻ",
        "session_id": "user123"  // Optional, sẽ tạo tự động nếu không có
    }
    """
    try:
        # Lấy query và session_id
        if request.method == 'GET':
            user_query = request.query_params.get('query', '')
            session_id = request.query_params.get('session_id', None)
        else:  # POST
            user_query = request.data.get('query', '')
            session_id = request.data.get('session_id', None)
        
        if not user_query:
            return Response({
                'success': False,
                'error': 'Query parameter is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # Tạo session_id nếu không có (cho user mới)
        if not session_id:
            session_id = str(uuid.uuid4())
            print(f"🆕 Generated new session_id: {session_id}")
        
        # Chat với Gemini (có chat history)
        result = rag_service.chat(user_query, session_id=session_id)
        
        return Response({
            'success': True,
            'data': {
                'query': result['query'],
                'answer': result['answer'],
                'tours': result['tours'],
                'session_id': result['session_id'],
                # 'chat_history': result['chat_history']  # Uncomment nếu muốn trả history
            }
        }, status=status.HTTP_200_OK)
        
    except Exception as e:
        print(f"Error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['POST'])
def clear_chat_history(request):
    """
    API endpoint để xóa chat history của user
    
    POST: /api/chat/clear
    Body: {"session_id": "user123"}
    """
    try:
        session_id = request.data.get('session_id')
        
        if not session_id:
            return Response({
                'success': False,
                'error': 'session_id is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        success = rag_service.clear_chat_session(session_id)
        
        if success:
            return Response({
                'success': True,
                'message': f'Chat history cleared for session {session_id}'
            }, status=status.HTTP_200_OK)
        else:
            return Response({
                'success': False,
                'error': f'No chat session found for {session_id}'
            }, status=status.HTTP_404_NOT_FOUND)
            
    except Exception as e:
        print(f"Error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


@api_view(['GET'])
def get_chat_history(request):
    """
    API endpoint để lấy chat history
    
    GET: /api/chat/history?session_id=user123
    """
    try:
        session_id = request.query_params.get('session_id')
        
        if not session_id:
            return Response({
                'success': False,
                'error': 'session_id is required'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        history = rag_service.get_chat_history(session_id)
        
        return Response({
            'success': True,
            'data': {
                'session_id': session_id,
                'history': history
            }
        }, status=status.HTTP_200_OK)
            
    except Exception as e:
        print(f"Error: {e}")
        return Response({
            'success': False,
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
