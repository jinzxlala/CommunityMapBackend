"""
URL configuration for map project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from myapp.views import LocationListCreateView
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth.models import User
from myapp.models import Location
from myapp.serializers import LocationGeoJSONSerializer
from myapp.views import UserFavoritesView
from myapp.views import LandmarkRecognitionView

class UserFavoritesView(APIView):
    def get(self, request):
        # 从 Cookie 获取用户 ID（假设前端存储了 user_id）
        user_id = request.COOKIES.get('user_id')
        if not user_id:
            return Response({"error": "User not authenticated"}, status=status.HTTP_401_UNAUTHORIZED)

        try:
            user = User.objects.get(id=user_id)
            favorites = user.favorite_locations.all()  # 获取用户收藏的地标
            serializer = LocationGeoJSONSerializer(favorites, many=True)
            return Response(serializer.data)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

from django.http import HttpResponse

def hello_view(request):
    return HttpResponse("hello")

urlpatterns = [
    path('admin/', admin.site.urls),
    path('hello/', hello_view),
    path('', include('myapp.urls')),
    path('api/', include('myapp.urls')),
    path('locations/', LocationListCreateView.as_view(), name='location-list'),
    path('api/favorites/', UserFavoritesView.as_view(), name='user-favorites'),
    path('api/recognize/', LandmarkRecognitionView.as_view(), name='landmark-recognition')
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)