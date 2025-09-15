from image_detection_package import get_landmark

res = get_landmark(r"D:\workshop\map_project\map\testimg1.jpg")
print(res)  # {'success': True, 'landmark': '...'}