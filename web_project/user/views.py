from django.shortcuts import render, redirect
from .forms import SignupForm, LoginForm
from django.contrib.auth import authenticate, login, logout
from django.urls import reverse
# Create your views here.

from django.shortcuts import render, get_object_or_404
from django.core.mail import send_mail
from .models import Exercise, WorkoutRoutine, ExerciseInRoutine


def profile(request):
    user_profile = get_object_or_404(UserProfile, user=request.user)
    return render(request, 'profile.html', {'user_profile': user_profile})


def select_exercises(request):
    exercises = Exercise.objects.all()
    return render(request, 'select_exercises.html', {'exercises': exercises})
    
def create_workout_routine(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        description = request.POST.get('description')
        exercises = request.POST.getlist('exercises')
        workout_routine = WorkoutRoutine.objects.create(name=name, description=description)
        for exercise_id in exercises:
            exercise = get_object_or_404(Exercise, pk=exercise_id)
            sets = request.POST.get('sets_%s' % exercise_id)
            reps = request.POST.get('reps_%s' % exercise_id)
            rest_time = request.POST.get('rest_time_%s' % exercise_id)
            ExerciseInRoutine.objects.create(exercise=exercise, workout_routine=workout_routine, sets=sets, reps=reps, rest_time=rest_time)
        # send email to user with workout routine
        send_mail(
            'Workout Routine',
            'Here is your workout routine: %s' % workout_routine,
            'from@example.com',
            ['to@example.com'],
            fail_silently=False,
        )
        return render(request, 'workout_routine_created.html', {'workout_routine': workout_routine})
    else:
        
        return redirect('select_exercises')



def register(request):
    context = {}
    context['login_form'] = LoginForm()
    context['signup_form'] = SignupForm()
    if request.method == "POST":
        if request.POST.get('submit') == 'login':
            if request.user.is_authenticated:
                return redirect('')
            if request.POST:
                form = LoginForm(request.POST)
                if form.is_valid():
                    phone = request.POST['phone']
                    password = request.POST['password']
                    user = authenticate(phone=phone, password=password)
                    if user:
                        login(request, user)
                        return redirect(reverse('homepage'))
                else:
                    context['login_form'] = form

            return render(request, 'register.html', context)
        elif request.POST.get('submit') == 'signup':
            print("xxxxxx")
            if request.POST:
                print(request.POST)
                
                form = SignupForm(request.POST)
                for field in form:
                    print("Field Error:", field.name,  field.errors)
                if form.is_valid():
                    print("ppppppp")
                    form.save()
                    phone = form.cleaned_data.get('phone')
                    raw_pass = form.cleaned_data.get('password1')
                    new_account = authenticate(phone=phone, password=raw_pass)
                    login(request, new_account)
                    return redirect(reverse('homepage'))

                else:
                    print("ffffffff")
                    context['signup_form'] = form
    print("heellooo")
    return render(request, 'register.html', context)


def usrlogout(request):
    logout(request)
    return redirect('/')

# def profile(request):
    
#     return render(request, 'user/profile.html')