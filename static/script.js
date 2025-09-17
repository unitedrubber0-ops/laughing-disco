document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('drawing-file');
    const fileNameDisplay = document.getElementById('file-name');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultsContainer = document.getElementById('results-container');
    const errorMessage = document.getElementById('error-message');

    // Display the name of the selected file
    fileInput.addEventListener('change', () => {
        fileNameDisplay.textContent = fileInput.files.length > 0 ? fileInput.files[0].name : '';
    });

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        if (fileInput.files.length === 0) {
            alert('Please select a PDF file to analyze.');
            return;
        }

        // Show loading spinner and hide previous results
        loadingSpinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');

        const formData = new FormData();
        formData.append('drawing', fileInput.files[0]);

        try {
            const response = await fetch('https://laughing-disco-docker.onrender.com/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const results = await response.json();
            displayResults(results);

        } catch (error) {
            displayError(error.message);
        } finally {
            // Hide loading spinner
            loadingSpinner.classList.add('hidden');
        }
    });

    function displayResults(data) {
        if (data.error) {
            displayError(data.error);
            return;
        }

        document.getElementById('result-part-number').textContent = data.part_number;
        document.getElementById('result-standard').textContent = data.standard;
        document.getElementById('result-grade').textContent = data.grade;
        document.getElementById('result-material').textContent = data.material;
        
        resultsContainer.classList.remove('hidden');
    }

    function displayError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
    }
});