# Generated by Django 3.0.5 on 2020-06-18 04:05

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('processing', '0006_remove_alg_project_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='alg',
            name='score',
            field=models.FloatField(default=1.1),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='alg',
            name='md_pk',
            field=models.CharField(max_length=200),
        ),
    ]
