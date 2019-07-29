from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=50)
    file = models.FileField(upload_to='')
    tag = models.CharField(max_length=100, blank=True)

    Private = 'Private'
    Public = 'Public'

    PRIVACY = [
        (Private, 'Private'),
        (Public, 'Public'),
    ]

    privacy = models.CharField(
            max_length=10,
            choices=PRIVACY,
            default=Private,
        )

    def __str__(self):
        return self.name
