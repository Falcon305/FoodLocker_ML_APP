from django.shortcuts import render, redirect
from sklearn.preprocessing import LabelEncoder
import numpy as np
from .models import Project
from .forms import DocumentForm
import pandas as pd
import io
import os
from django.contrib.auth.decorators import login_required
from sklearn.base import TransformerMixin

# Create your views here.
@login_required
def create_project(request):
        form = DocumentForm()
        return render(request, 'processing/create.html', {'form': form})

@login_required
def load_dataset(request):
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                title = request.POST['title']
                description = request.POST['description']
                newdoc = Project(dataset=request.FILES['dataset'], user=request.user, title=title, description=description)
                newdoc.save()
                file_name = request.FILES['dataset'].name
                df = pd.read_csv("media/user_{0}/raw_csv/{1}".format(request.user, file_name))
                buf = io.StringIO()
                df.info(buf=buf)
                s = buf.getvalue()
                lines = s.strip().split('\n')
                entries = lines[1].strip().split()[1]
                size = (' '.join(lines[-1].strip().split()[2:]))

                return render(request, 'processing/dataset.html', {'entries': entries,
                                                            'size': size,
                                                            'title': title,
                                                            'filename':file_name})
            except Exception as e:
                return render(request, 'processing/dataset.html', {"Error": e})
        else:
            return redirect('create_project')
    else:
        return redirect("index")