// Configure API URL based on environment
const API_BASE_URL = window.API_BASE_URL || 'https://laughing-disco-docker.onrender.com';
const LOCAL_API_URL = 'http://localhost:5000';

// Helper function to display errors with console logging
function displayErrorWithLogging(error, details = null) {
    console.error('Upload error:', error);
    if (details) console.error('Error details:', details);
    errorMessage.textContent = error.message || 'An unexpected error occurred. Please try again.';
    errorMessage.classList.remove('hidden');
    loadingSpinner.classList.add('hidden');
}

// Global UI elements
const loadingSpinner = document.getElementById('loading-spinner');
const errorMessage = document.getElementById('error-message');
const resultsContainer = document.getElementById('results-container');

// Error handling function
function displayErrorWithLogging(error, details = null) {
    console.error('Upload error:', error);
    if (details) console.error('Error details:', details);
    
    if (errorMessage) {
        errorMessage.textContent = error.message || 'An unexpected error occurred. Please try again.';
        errorMessage.classList.remove('hidden');
    }
    
    if (loadingSpinner) {
        loadingSpinner.classList.add('hidden');
    }
    
    // Hide results if there was an error
    if (resultsContainer) {
        resultsContainer.classList.add('hidden');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const fileInput = document.getElementById('drawing-file');
    const fileNameDisplay = document.getElementById('file-name');

    // Display the name of the selected file
    fileInput.addEventListener('change', () => {
        fileNameDisplay.textContent = fileInput.files.length > 0 ? fileInput.files[0].name : '';
    });

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();

        if (fileInput.files.length === 0) {
            displayErrorWithLogging(new Error('Please select a PDF file to analyze.'));
            return;
        }

        // Show loading spinner and hide previous results
        loadingSpinner.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        errorMessage.classList.add('hidden');

        const formData = new FormData();
        formData.append('drawing', fileInput.files[0]);

        try {
            const apiUrl = window.location.hostname === 'localhost' 
                ? `${LOCAL_API_URL}/api/analyze`
                : `${API_BASE_URL}/api/analyze`;
            
            console.log('Sending request to:', apiUrl); // Debug log
            
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData,
                mode: 'cors',
                credentials: 'include',
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
            }

            const results = await response.json();
            
            // Log the response for debugging
            console.log('API Response:', results);
            
            // Check for API-level errors
            if (results.error) {
                throw new Error(results.error);
            }
            
            // Check for empty results
            if (Object.values(results).every(v => !v || v === "Not Found")) {
                throw new Error("No data could be extracted from the PDF. Please ensure it's text-selectable.");
            }
            
            displayResults(results);

        } catch (error) {
            console.error('Request failed:', error);
            let errorMessage = error.message;
            
            // Enhance error messages based on error type
            if (error instanceof TypeError) {
                if (error.message.includes('NetworkError')) {
                    errorMessage = 'Unable to connect to the server. Please check your connection and try again.';
                } else if (error.message.includes('timeout')) {
                    errorMessage = 'The request timed out. Please try with a smaller PDF file.';
                } else {
                    errorMessage = 'Network error occurred. Please try again.';
                }
            }
            
            displayErrorWithLogging(new Error(errorMessage), error);
        } finally {
            // Hide loading spinner and reset form
            loadingSpinner.classList.add('hidden');
            fileInput.value = '';
            fileNameDisplay.textContent = '';
        }
    });

    function displayResults(data) {
        // Check for error or empty results
        if (data.error || Object.values(data).every(v => !v || v === "Not Found")) {
            const errorMessage = data.error || "No data could be extracted - please ensure the PDF is text-selectable";
            console.error('Analysis failed:', data);  // Log for debugging
            displayErrorWithLogging(new Error(errorMessage));
            return;
        }

        console.log('Displaying results:', data);  // Log for debugging
        resultsContainer.classList.remove('hidden');

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