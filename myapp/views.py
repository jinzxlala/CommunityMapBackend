from rest_framework import generics, permissions, filters
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, parsers
from django.conf import settings
from django_filters.rest_framework import DjangoFilterBackend
from myapp.models import Location
from myapp.serializers import LocationGeoJSONSerializer
from django.contrib.auth.models import User
import os
import requests
import base64
# import paddlehub as hub

class LandmarkRecognitionView(APIView):
    parser_classes = [parsers.MultiPartParser]  # 支持文件上传

    def post(self, request):
        # 1. 获取上传的图片
        uploaded_file = request.FILES.get('image')
        if not uploaded_file:
            return Response({"error": "未提供图片"}, status=status.HTTP_400_BAD_REQUEST)

        # 2. 保存图片到临时文件
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp', uploaded_file.name)
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        with open(temp_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        # 3. 调用地标识别模型（示例使用伪代码）
        try:
            # 替换为实际模型调用（如 PaddleHub、PyTorch）
            landmark_name, confidence = self._predict_landmark(temp_path)
            
            # 4. 返回识别结果
            return Response({
                "landmark": landmark_name,
                "confidence": confidence,
                "image_url": f"/media/temp/{uploaded_file.name}"
            })
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        finally:
            os.remove(temp_path)  # 清理临时文件

    def _predict_landmark(self, image_path):
        """调用地标识别模型（示例）"""
        # 实际项目中替换为你的模型代码（如 PaddleHub、PyTorch）
        # 返回地标名称和置信度
        return "埃菲尔铁塔", 0.95
class LocationListCreateView(generics.ListCreateAPIView):
    queryset = Location.objects.all()
    serializer_class = LocationGeoJSONSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [filters.SearchFilter, DjangoFilterBackend]
    search_fields = ['name']
    filterset_fields = ['name','owner__username']

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

class LocationDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Location.objects.all()
    serializer_class = LocationGeoJSONSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

class LocationListAPI(APIView):
    def get(self, request):
        locations = Location.objects.all()
        serializer = LocationGeoJSONSerializer(locations, many=True)
        return Response(serializer.data)
    def post(self, request):
        serializer = LocationGeoJSONSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class ImageUploadView(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    def post(self, request, format=None):
        # ... 原有代码 ...
        
        # 修改返回的图片URL为完整地址
        image_url = f"{settings.SITE_URL}{settings.MEDIA_URL}location_images/{file_obj.name}"
        
        return Response({
            'image_url': image_url,
            'recognition': result
        })

    def post(self, request, format=None):
        file_obj = request.FILES.get('image')
        if not file_obj:
            return Response({'error': 'No image uploaded.'}, status=status.HTTP_400_BAD_REQUEST)
        # 保存图片到media/location_images/
        save_path = os.path.join(settings.MEDIA_ROOT, 'location_images', file_obj.name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb+') as destination:
            for chunk in file_obj.chunks():
                destination.write(chunk)
        image_url = settings.MEDIA_URL + 'location_images/' + file_obj.name
        result = recognize_image(save_path)
        return Response({'image_url': image_url, 'recognition': result})

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


def recognize_image(image_path):
    from image_detection_package.recognition_dispatcher import get_landmark
    return get_landmark(image_path)
