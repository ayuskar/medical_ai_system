from django import forms
from .models import Review

class ReviewForm(forms.ModelForm):
    class Meta:
        model = Review
        fields = ['rating', 'review_text']
        widgets = {
            'rating': forms.RadioSelect(attrs={
                'class': 'flex space-x-2'
            }),
            'review_text': forms.Textarea(attrs={
                'rows': 4,
                'class': 'w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500',
                'placeholder': 'Share your experience with the doctor...'
            }),
        }