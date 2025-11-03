from image_detection_package import get_landmark

res = get_landmark(r"D:\workshop\map_project\CommunityMapBackend\test2.jpg")
print(res)  # {'success': True, 'landmark': '...'}