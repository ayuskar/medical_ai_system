from django.core.management.base import BaseCommand
from contact.models import ContactCategory, FAQ

class Command(BaseCommand):
    help = 'Create initial contact categories and FAQs'
    
    def handle(self, *args, **options):
        # Create contact categories
        categories = [
            ('general', 'General questions about our services'),
            ('technical', 'Technical issues with the platform'),
            ('billing', 'Billing and payment questions'),
            ('medical', 'Medical-related inquiries'),
            ('feedback', 'Feedback and suggestions'),
            ('complaint', 'Complaints about service'),
            ('partnership', 'Partnership opportunities'),
            ('other', 'Other types of inquiries'),
        ]
        
        for name, description in categories:
            ContactCategory.objects.get_or_create(
                name=name,
                defaults={'description': description}
            )
        
        # Create sample FAQs
        faqs = [
            {
                'question': 'How do I book an appointment?',
                'answer': 'You can book an appointment by clicking on "Find Doctors" in the navigation, selecting a doctor, and choosing an available time slot.',
                'category': 'general',
                'order': 1
            },
            {
                'question': 'Is my medical information secure?',
                'answer': 'Yes, we use industry-standard encryption and follow HIPAA guidelines to protect your medical information.',
                'category': 'privacy',
                'order': 1
            },
            # Add more FAQs as needed
        ]
        
        for faq_data in faqs:
            FAQ.objects.get_or_create(
                question=faq_data['question'],
                defaults=faq_data
            )
        
        self.stdout.write(
            self.style.SUCCESS('Successfully created contact categories and FAQs')
        )