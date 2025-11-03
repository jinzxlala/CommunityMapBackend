from django.urls import include,path
from .views import (
    LocationListCreateView, LocationDetailView, ImageUploadView, 
    LocationListAPI, LandmarksView, LandmarkItemView, MapLocationSyncView,
    LoginView, LogoutView, CheckAuthView
)

urlpatterns = [
    # 兼容原有 locations API
    path('locations/', LocationListAPI.as_view(), name='location-list'),
    path('locations/<int:pk>/', LocationDetailView.as_view(), name='location-detail'),

    # 前端期望的 landmarks 路径
    path('landmarks', LandmarksView.as_view(), name='landmarks'),
    path('landmarks/<int:pk>', LandmarkItemView.as_view(), name='landmark-item'),

    # 图片上传
    path('upload-image/', ImageUploadView.as_view(), name='upload-image'),

    # 同步接口
    path('map/location', MapLocationSyncView.as_view(), name='map-location-sync'),

    # 认证接口
    path('auth/login', LoginView.as_view(), name='login'),
    path('auth/logout', LogoutView.as_view(), name='logout'),
    path('auth/check', CheckAuthView.as_view(), name='check-auth'),
]