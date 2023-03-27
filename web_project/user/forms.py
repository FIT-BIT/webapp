from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.db.models import fields
from django.contrib.auth import authenticate
from .models import User
from django.utils.translation import gettext_lazy as _


class SignupForm(UserCreationForm):
    phone = forms.CharField(help_text="A valid phone no. id is required")
    # phone = PhoneNumberField()

    class Meta:
        model = User
        fields = ('phone', 'first_name', 'last_name', 'isGymTrainer', 'isPhysiotherapist', 'password1', 'password2')
        labels = {
            'isGymTrainer': _('Do you want to register as a Gym trainer?'),
            'isPhysiotherapist': _('Do you want to register as a Physiotherapist?'),
        }


class LoginForm(forms.ModelForm):
    password = forms.CharField(label='Password', widget=forms.PasswordInput)

    class Meta:
        model = User
        fields = ('phone', 'password')

    def clean(self):
        if self.is_valid():
            phone = self.cleaned_data['phone']
            password = self.cleaned_data['password']
            if not authenticate(phone=phone, password=password):
                raise forms.ValidationError("Invalid Login Credentials")
