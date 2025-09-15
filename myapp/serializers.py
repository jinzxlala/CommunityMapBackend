# myapp/serializers.py
import json
from rest_framework import serializers
from .models import Location, Article

class LocationSerializer(serializers.ModelSerializer):
    owner = serializers.ReadOnlyField(source='owner.username')
    image = serializers.ImageField(required=False)
    class Meta:
        model = Location
        fields = ['id', 'name', 'latitude', 'longitude', 'description', 'image', 'owner']

class ArticleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Article
        fields = ['id', 'title', 'content', 'pub_date']

class LocationGeoJSONSerializer(serializers.ModelSerializer):
    class Meta:
        model = Location
        fields = '__all__'
    def to_representation(self, instance):
        image_url = None
        if instance.image:
            image_url = f"{settings.SITE_URL}{instance.image.url}"
        return {
            "type": "Feature",
            "properties": {
                "id": instance.id,
                "name": instance.name,
                "description": instance.description,
                "image": image_url,
                "owner": instance.owner.username
            },
            "geometry": {
                "type": "Point",
                "coordinates": [
                    instance.longitude,
                    instance.latitude
                ]
            }
        }

