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
            const apiUrl = process.env.NODE_ENV === 'production'
                ? 'https://feasibility-analyzer-api.onrender.com/api/analyze'
                : 'http://localhost:5000/api/analyze';
            
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData,
                mode: 'cors',
                credentials: 'same-origin',
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

        // Update main results
        document.getElementById('result-child-part').textContent = data.child_part;
        document.getElementById('result-description').textContent = data.description;
        document.getElementById('result-specification').textContent = data.specification;
        document.getElementById('result-material').textContent = data.material;
        document.getElementById('result-od').textContent = data.od;
        document.getElementById('result-thickness').textContent = data.thickness;
        document.getElementById('result-centerline').textContent = data.centerline_length;
        document.getElementById('result-development').textContent = data.development_length_mm;
        document.getElementById('result-burst-pressure').textContent = data.burst_pressure_bar;

        // Display coordinates if available
        const coordsContainer = document.getElementById('coordinates-container');
        const coordsBody = document.getElementById('coordinates-body');
        
        if (data.coordinates && data.coordinates.length > 0) {
            coordsBody.innerHTML = ''; // Clear existing coordinates
            data.coordinates.forEach((coord, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>P${index}</td>
                    <td>${coord.x.toFixed(2)}</td>
                    <td>${coord.y.toFixed(2)}</td>
                    <td>${coord.z.toFixed(2)}</td>
                `;
                coordsBody.appendChild(row);
            });
            coordsContainer.classList.remove('hidden');
        } else {
            coordsContainer.classList.add('hidden');
        }
        
        resultsContainer.classList.remove('hidden');
    }

    function displayError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
    }
});