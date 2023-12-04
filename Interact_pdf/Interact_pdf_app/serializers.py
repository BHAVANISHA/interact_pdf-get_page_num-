from rest_framework import serializers
from Interact_pdf_app.models import UploadedFile

class UploadedFileSerializer( serializers.ModelSerializer ):
    class Meta:
        model=UploadedFile
        fields="__all__"
