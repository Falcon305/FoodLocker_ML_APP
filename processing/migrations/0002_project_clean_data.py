# Generated by Django 3.0.5 on 2020-04-10 16:29

from django.db import migrations, models
import processing.models


class Migration(migrations.Migration):

    dependencies = [
        ('processing', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='clean_data',
            field=models.FileField(blank=True, upload_to=processing.models.user_directory_path),
        ),
    ]
