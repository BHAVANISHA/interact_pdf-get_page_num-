from django.db import models


class UploadedFile( models.Model ):
    file_upload = models.FileField()
    query = models.CharField( max_length=200 )
