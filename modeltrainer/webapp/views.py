from django.shortcuts import render
from django.contrib.auth.decorators import login_required
from django import forms

def homepage(request):
    return render(request, 'webs/index.html')

@login_required
def profile(request):
    user = request.user
    return render(request, 'webs/profile.html', {'user': user})

@login_required
def settings(request):
    user = request.user
    return render(request, 'webs/settings.html', {'user': user})

def about(request):
    return render(request, 'webs/about.html')

def contact(request):
    return render(request, 'webs/contact.html')
