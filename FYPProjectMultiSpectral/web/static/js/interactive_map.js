// static/js/interactive_map.js
document.addEventListener('DOMContentLoaded', function() {
    // Initialize the map
    var map = L.map('map').setView([35.9375, 14.3754], 11); // Malta
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles © Esri',
        maxZoom: 19
    }).addTo(map);

    setTimeout(function() {
        map.invalidateSize();
    }, 100);

    var imageOverlay = null;
    var imageContainer = document.getElementById('image-container');
    var predictionContainer = document.getElementById('prediction-container');

    function fetchImageAndPrediction(lat, lon) {
        var formData = new FormData();
        formData.append('lat', lat);
        formData.append('lon', lon);

        // Clear previous results and show loading indicators
        if (imageOverlay) {
            map.removeLayer(imageOverlay);
            imageOverlay = null;
        }
        imageContainer.innerHTML = `
            <div class="card shadow-sm">
                <div class="card-body text-center">
                    <p><i class="fas fa-spinner fa-spin"></i> Loading image...</p>
                </div>
            </div>`;
        predictionContainer.innerHTML = `
            <div class="card shadow-sm">
                <div class="card-body text-center">
                    <p><i class="fas fa-spinner fa-spin"></i> Loading prediction...</p>
                </div>
            </div>`;

        // Step 1: Fetch the image patch
        fetch('/get_image', {
            method: 'POST',
            body: formData,
            headers: { 'Accept': 'application/json' }
        })
        .then(response => {
            console.log('get_image response:', response.status, response.ok);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('get_image data:', data);
            imageOverlay = L.imageOverlay(data.image_url, data.bounds).addTo(map);
            map.fitBounds(data.bounds);

            imageContainer.innerHTML = `
                <div class="card shadow-sm">
                    <div class="card-header bg-primary text-white fw-bold fs-4">Selected Patch</div>
                    <div class="card-body">
                        <img src="${data.image_url}" alt="RGB Image Patch" class="img-fluid mb-3 rounded" style="width: 300px;">
                        <p class="card-text"><small class="text-muted">Multispectral TIFF saved as: ${data.tiff_file}</small></p>
                        <p class="card-text"><small class="text-muted">Coords: ${lat.toFixed(4)}, ${lon.toFixed(4)}</small></p>
                    </div>
                </div>`;

            // Step 2: Fetch predictions
            var experimentSelect = document.getElementById('experiment-select');
            var selectedExperiment = experimentSelect.value;

            if (!selectedExperiment) {
                predictionContainer.innerHTML = `<div class="alert alert-warning">Please select an experiment before clicking on the map.</div>`;
                return;
            }
            var predictFormData = new FormData();
            predictFormData.append('experiment', selectedExperiment);

            return fetch('/predict_from_map', {
                method: 'POST',
                body: predictFormData,
                headers: { 'Accept': 'application/json' }
            });
        })
        .then(response => {
            console.log('[predict_from_map] Status:', response.status, 'OK:', response.ok);
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`[predict_from_map] HTTP error! status: ${response.status}, body: ${text}`);
                });
            }
            return response.json(); // Parse as JSON, not text
        })
        .then(data => {
            console.log('predict_from_map data:', data);
            if (data.error) {
                predictionContainer.innerHTML = `<div class="alert alert-danger">Error: ${data.error}</div>`;
                return;
            }

            // Render predictions
            let predictionHtml = `
                <div class="card shadow-sm">
                    <div class="card-header bg-success text-white fw-bold fs-4">Prediction Results</div>
                    <div class="card-body">
                        <h5>Experiment: ${data.selected_experiment}</h5>
                        <table class="table table-striped">
                            <thead>
                                <tr>
                                    <th>Land Cover</th>
                                    <th>Probability</th>
                                </tr>
                            </thead>
                            <tbody>`;
            data.predictions[data.selected_experiment].forEach(pred => {
                predictionHtml += `
                    <tr>
                        <td>${pred.label}</td>
                        <td>${(pred.probability * 100).toFixed(2)}%</td>
                    </tr>`;
            });
            predictionHtml += `
                            </tbody>
                        </table>
                        <h5 class="fw-bold">Grad-CAM Visualizations</h5>
                        <div class="row">`;
            for (let [className, url] of Object.entries(data.gradcam)) {
                predictionHtml += `
                    <div class="col-md-6">
                        <img src="${url}" alt="Grad-CAM for ${className}" class="img-fluid rounded mb-2">
                        <p class="text-center">${className}</p>
                    </div>`;
            }
            predictionHtml += `
                        </div>
                        <h5 class="fw-bold">Color-Coded Grad-CAM</h5>
                        <img src="${data.gradcam_colorcoded_.combined.url}" alt="Color-Coded Grad-CAM" class="img-fluid rounded mb-2" style="width: 300px;">
                        <div class="legend-title">Legend</div>
                        <div class="legend-container">`;
            
            // Render the legend using CSS classes
            for (let [category, color] of Object.entries(data.gradcam_colorcoded_.combined.legend)) {
                predictionHtml += `
                    <div class="legend-item">
                        <div class="legend-color-box" style="background-color: ${color};"></div>
                        <span class="legend-text">${category}</span>
                    </div>`;
            }
            predictionHtml += `
                        </div>
                    </div>
                </div>`;
            predictionContainer.innerHTML = predictionHtml;
        })
        .catch(error => {
            console.error('Error during fetch process:', error);
            const errorMsg = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            imageContainer.innerHTML = errorMsg;
            predictionContainer.innerHTML = errorMsg;
        });
    }

    map.on('click', function(e) {
        var lat = e.latlng.lat;
        var lon = e.latlng.lng;
        fetchImageAndPrediction(lat, lon);
    });
});