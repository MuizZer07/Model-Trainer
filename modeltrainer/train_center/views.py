from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from .models import Dataset
from modeltrainer import settings
from django.contrib import messages
from modelenv.model import Run_Model

@login_required
def upload_dataset(request):
    user = request.user
    return render(request, 'train/upload_dataset.html', {'user': user})

@login_required
def upload_dataset_request(request):
    if  request.method == 'POST':
        dataset = request.FILES['data']
        fs = FileSystemStorage()
        filename = fs.save(dataset.name, dataset)

        new_dataset = Dataset()
        new_dataset.owner = request.user
        new_dataset.name = request.POST.get('dataset_name')
        new_dataset.file = filename
        new_dataset.tag = request.POST.get('tag')
        new_dataset.privacy = request.POST.get('privacy')
        new_dataset.save()

        messages.add_message(request, messages.INFO, 'Dataset has been uploaded successfully!')
    else:
        messages.add_message(request, messages.INFO, 'Error occured while uploading dataset!')
    return redirect('profile')

@login_required
def train_model(request):
    user = request.user
    datasets = Dataset.objects.filter(owner=user)

    training = ['Machine Learning', 'Convolutional Neural Networks', 'Transfer Learning' ]

    training_options = {
        'Transfer Learning': ['resnet', 'alexnet', 'vgg', 'squeezenet', 'densenet', 'inception'],
        'Convolutional Neural Networks': [],
        'Machine Learning': [],
    }


    return render(request, 'train/train_model.html', {
                'user': user, 'datasets':datasets,
                'training':training,
                'training_options': training_options
    })

@login_required
def train_model_request(request):
    if  request.method == 'POST':
        model_name = request.POST.get('training_model')
        dataset_url = settings.MEDIA_ROOT + '/' + request.POST.get('dataset_url')

        model_train = Run_Model(request.user, model_name, dataset_url)
        val_acc = model_train.get_results()
        
        messages.add_message(request, messages.INFO, val_acc)
    return redirect('profile')
