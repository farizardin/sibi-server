from django import forms
from django.core.validators import FileExtensionValidator

class FileForm(forms.Form):
    video = forms.FileField(validators=[FileExtensionValidator(allowed_extensions=['mp4'])])