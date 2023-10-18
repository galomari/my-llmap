from django.db import models

class TrainingHistory(models.Model):
    training_date = models.DateTimeField(auto_now_add=True)
    accuracy = models.FloatField()
    loss = models.FloatField()
    validation_accuracy = models.FloatField()
    validation_loss = models.FloatField()
    epoch_duration = models.DurationField()
    total_duration = models.DurationField()
    model_path = models.FileField(upload_to=r'C:\Users\hp\Downloads\models')
    
    def __str__(self):
        return f"Training on {self.training_date}"
