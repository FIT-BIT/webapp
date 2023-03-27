from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from django.db.models.deletion import CASCADE
from django.db.models.fields.related import ManyToManyField
from django.core.validators import RegexValidator
# Create your models here.
# from mongoengine import Document,fields


class MyUserManager(BaseUserManager):
    def create_user(self, phone, first_name, last_name, isGymTrainer, isPhysiotherapist, isTrainee, password=None):
        if not phone:
            return ValueError("Farmer must enter contact no.")

        if not first_name:
            return ValueError("Farmer must enter first name")

        if not last_name:
            return ValueError("Farmer must enter last name")

        user = self.model(
            phone=phone,
            first_name=first_name,
            last_name=last_name,
            isGymTrainer=isGymTrainer,
            isPhysiotherapist=isPhysiotherapist,
            isTrainee=isTrainee,
        )
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, phone, first_name, last_name, isGymTrainer, isPhysiotherapist, isTrainee, password=None):
        user = self.create_user(
            phone=phone,
            first_name=first_name,
            last_name=last_name,
            isGymTrainer=isGymTrainer,
            isPhysiotherapist=isPhysiotherapist,
            isTrainee=isTrainee,
            password=password
        )
        user.is_admin = True
        user.is_staff = True
        user.is_superuser = True
        user.save(using=self._db)


class User(AbstractBaseUser):
    username = None
    # phone = models.CharField(max_length=20,unique=True, primary_key=True)
    # phone = PhoneNumberField(null=False, blank=False, unique=True, primary_key=True)
    phone_regex = RegexValidator(
        regex=r'^\+91\d{10}$', message="Enter a valid 10 digit number with country code(+91).")
    phone = models.CharField(validators=[
                             phone_regex], max_length=17, null=False, blank=False, unique=True, primary_key=True)
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    # location = models.CharField(max_length=200)
    isGymTrainer = models.BooleanField(default=False)
    isPhysiotherapist = models.BooleanField(default=False)
    isTrainee = models.BooleanField(default=False)

    date_joined = models.DateTimeField(auto_now_add=True)
    last_login = models.DateTimeField(auto_now=True)
    is_admin = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    USERNAME_FIELD = 'phone'
    REQUIRED_FIELDS = ['first_name', 'last_name', 'isGymTrainer', 'isPhysiotherapist', 'isTrainee']

    objects = MyUserManager()

    def __str__(self):
        return self.phone

    def has_perm(self, perm, obj=None):
        return self.is_admin

    def has_module_perms(self, app_label):
        return True
