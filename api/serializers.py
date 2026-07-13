from rest_framework import serializers
from django.contrib.auth.models import User
from .models import OperatorProfile

class UserSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True)

    class Meta:
        model = User
        fields = ('username', 'password')

    def create(self, validated_data):
        # 1. Securely create the user
        user = User.objects.create_user(
            username=validated_data['username'],
            password=validated_data['password']
        )
        # 2. Automatically generate their profile with a default $5000 budget
        OperatorProfile.objects.create(user=user, monthly_budget_limit=5000.00)
        return user