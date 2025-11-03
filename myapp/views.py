from rest_framework import generics, permissions, filters
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, parsers
from rest_framework.authtoken.models import Token
from django.conf import settings
from django_filters.rest_framework import DjangoFilterBackend
from myapp.models import Location
from myapp.serializers import LocationGeoJSONSerializer, LocationFlatSerializer
from django.contrib.auth import authenticate
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
    serializer_class = LocationFlatSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [filters.SearchFilter, DjangoFilterBackend]
    search_fields = ['name']
    filterset_fields = ['name','owner__username']

    def perform_create(self, serializer):
        serializer.save(owner=self.request.user)

class LocationDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Location.objects.all()
    serializer_class = LocationFlatSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]

    def delete(self, request, *args, **kwargs):
        instance = self.get_object()
        instance.delete()
        return Response({"success": True})

class LocationListAPI(APIView):
    def get(self, request):
        locations = Location.objects.all()
        serializer = LocationFlatSerializer(locations, many=True)
        return Response({"success": True, "data": serializer.data})
    def post(self, request):
        serializer = LocationFlatSerializer(data=request.data)
        if serializer.is_valid():
            location = serializer.save()
            return Response({"success": True, "data": LocationFlatSerializer(location).data}, status=status.HTTP_201_CREATED)
        return Response({"success": False, "message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

class ImageUploadView(APIView):
    permission_classes = [permissions.AllowAny]
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
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
        
        # 如果识别成功，尝试在数据库中匹配地标
        matched_location = None
        if result.get('success') and result.get('landmark'):
            landmark_name = result.get('landmark')
            # 在数据库中查找匹配的地标（精确匹配或模糊匹配）
            try:
                # 先尝试精确匹配
                matched_location = Location.objects.filter(name=landmark_name).first()
                # 如果没有精确匹配，尝试模糊匹配
                if not matched_location:
                    matched_location = Location.objects.filter(name__icontains=landmark_name).first()
            except Exception as e:
                print(f"数据库匹配错误: {e}")
        
        # 构建返回结果
        response_data = {
            'image_url': image_url,
            'recognition': result
        }
        
        # 如果找到匹配的地标，添加地标详细信息
        if matched_location:
            response_data['matched_location'] = {
                'id': matched_location.id,
                'name': matched_location.name,
                'description': matched_location.description,
                'category': matched_location.category,
                'latitude': matched_location.latitude,
                'longitude': matched_location.longitude
            }
        
        return Response(response_data)

class LandmarksView(APIView):
    permission_classes = [permissions.AllowAny]

    def get(self, request):
        """获取地标列表 - 所有用户可访问"""
        locations = Location.objects.all()
        data = LocationFlatSerializer(locations, many=True).data
        return Response({"success": True, "data": data})

    def post(self, request):
        """创建地标 - 仅管理员可访问"""
        # 检查用户是否为管理员
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response({
                "success": False, 
                "message": "需要管理员权限"
            }, status=status.HTTP_403_FORBIDDEN)
        
        serializer = LocationFlatSerializer(data=request.data)
        if serializer.is_valid():
            location = serializer.save()
            return Response({"success": True, "data": LocationFlatSerializer(location).data}, status=status.HTTP_201_CREATED)
        return Response({"success": False, "message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

class LandmarkItemView(APIView):
    permission_classes = [permissions.AllowAny]

    def put(self, request, pk):
        """更新地标 - 仅管理员可访问"""
        # 检查用户是否为管理员
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response({
                "success": False, 
                "message": "需要管理员权限"
            }, status=status.HTTP_403_FORBIDDEN)
        
        try:
            location = Location.objects.get(pk=pk)
        except Location.DoesNotExist:
            return Response({"success": False, "message": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        
        serializer = LocationFlatSerializer(location, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            return Response({"success": True, "data": serializer.data})
        return Response({"success": False, "message": serializer.errors}, status=status.HTTP_400_BAD_REQUEST)

    def delete(self, request, pk):
        """删除地标 - 仅管理员可访问"""
        # 检查用户是否为管理员
        if not request.user.is_authenticated or not request.user.is_staff:
            return Response({
                "success": False, 
                "message": "需要管理员权限"
            }, status=status.HTTP_403_FORBIDDEN)
        
        try:
            location = Location.objects.get(pk=pk)
        except Location.DoesNotExist:
            return Response({"success": False, "message": "Not found"}, status=status.HTTP_404_NOT_FOUND)
        location.delete()
        return Response({"success": True})

class MapLocationSyncView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request):
        # 接受 GeoJSON Feature 数组并返回 success
        try:
            payload = request.data
            if isinstance(payload, list):
                return Response({"success": True, "received": len(payload)})
            return Response({"success": True})
        except Exception as e:
            return Response({"success": False, "message": str(e)}, status=status.HTTP_400_BAD_REQUEST)

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
    from image_detection_package import get_landmark
    return get_landmark(image_path)

class LoginView(APIView):
    """用户登录接口"""
    permission_classes = [permissions.AllowAny]
    
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        
        if not username or not password:
            return Response({
                'success': False,
                'message': '用户名和密码不能为空'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 验证用户
        user = authenticate(username=username, password=password)
        
        if user is None:
            return Response({
                'success': False,
                'message': '用户名或密码错误'
            }, status=status.HTTP_401_UNAUTHORIZED)
        
        # 创建或获取token
        token, created = Token.objects.get_or_create(user=user)
        
        return Response({
            'success': True,
            'message': '登录成功',
            'data': {
                'token': token.key,
                'user_id': user.id,
                'username': user.username,
                'is_staff': user.is_staff,
                'is_superuser': user.is_superuser
            }
        })

class LogoutView(APIView):
    """用户登出接口"""
    permission_classes = [permissions.IsAuthenticated]
    
    def post(self, request):
        try:
            # 删除用户的token
            request.user.auth_token.delete()
            return Response({
                'success': True,
                'message': '登出成功'
            })
        except Exception as e:
            return Response({
                'success': False,
                'message': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class CheckAuthView(APIView):
    """检查用户认证状态"""
    permission_classes = [permissions.IsAuthenticated]
    
    def get(self, request):
        return Response({
            'success': True,
            'data': {
                'user_id': request.user.id,
                'username': request.user.username,
                'is_staff': request.user.is_staff,
                'is_superuser': request.user.is_superuser
            }
        })
