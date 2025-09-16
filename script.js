document.addEventListener('DOMContentLoaded', function() {
    const contactForm = document.getElementById('contact-form');
    const formStatus = document.getElementById('form-status');

    contactForm.addEventListener('submit', async function(event) {
        event.preventDefault(); // Stop the default form submission

        // Show a "processing" message to the user
        formStatus.textContent = 'Analyzing your message...';
        formStatus.style.color = '#555';

        // Get data from the form
        const formData = new FormData(contactForm);
        const name = formData.get('name');
        const message = formData.get('message');

        try {
            // Send the data to our Python backend API endpoint
            const response = await fetch('http://127.0.0.1:5000/api/contact', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ name: name, message: message }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            // Get the AI-generated reply from the backend
            const data = await response.json();
            
            // Display the smart reply to the user!
            formStatus.textContent = data.reply;
            formStatus.style.color = 'green';

        } catch (error) {
            console.error('Error:', error);
            // Show a generic fallback message if something went wrong
            formStatus.textContent = 'Thank you for your message! We will get back to you soon.';
            formStatus.style.color = 'green';
        }
    });
});