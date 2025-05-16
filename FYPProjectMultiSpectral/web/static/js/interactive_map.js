// Interactive map for displaying image patches and predictions

document.addEventListener('DOMContentLoaded', function() {
    // Initialise the map
    var map = L.map('map').setView([35.9375, 14.3754], 11); // Malta
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles © Esri',
        maxZoom: 19
    }).addTo(map);

    setTimeout(function() {
        map.invalidateSize();
    }, 100);

    var imageOverlay = null;
    var previewRectangle = null;
    var imageContainer = document.getElementById('image-container');
    var predictionContainer = document.getElementById('prediction-container');

    // Function to calculate square bounds in Web Mercator projection
    function calculateSquareBounds(lat, lon) {
        // Step 1: Convert the center point to Web Mercator coordinates
        const centerPoint = map.project([lat, lon], map.getZoom());

        // Step 2: Calculate the size of the patch in Web Mercator units
        const zoom = map.getZoom();
        const metersPerPixel = 40075016.686 / (256 * Math.pow(2, zoom)); // Earth's circumference / pixels at zoom
        const patchSizeMeters = 1200; // 1200m x 1200m
        const patchSizePixels = patchSizeMeters / metersPerPixel; // Size in Web Mercator units

        // Step 3: Calculate the square bounds in Web Mercator coordinates
        const halfSize = patchSizePixels / 2;
        const southWestPoint = L.point(centerPoint.x - halfSize, centerPoint.y + halfSize);
        const northEastPoint = L.point(centerPoint.x + halfSize, centerPoint.y - halfSize);

        // Step 4: Convert back to geographic coordinates
        const southWestLatLng = map.unproject(southWestPoint, zoom);
        const northEastLatLng = map.unproject(northEastPoint, zoom);

        return [
            [southWestLatLng.lat, southWestLatLng.lng], // South-West corner
            [northEastLatLng.lat, northEastLatLng.lng]  // North-East corner
        ];
    }

    // Function to draw the preview rectangle
    function drawPreviewRectangle(lat, lon) {
        if (previewRectangle) {
            map.removeLayer(previewRectangle);
        }

        const bounds = calculateSquareBounds(lat, lon);

        previewRectangle = L.rectangle(bounds, {
            color: '#fff',
            weight: 2,
            opacity: 0.8,
            fillOpacity: 0,
            dashArray: '5, 5'
        }).addTo(map);
    }

    // Function to remove the preview rectangle
    function removePreviewRectangle() {
        if (previewRectangle) {
            map.removeLayer(previewRectangle);
            previewRectangle = null;
        }
    }

    // Function to fetch the image and predictions
    function fetchImageAndPrediction(lat, lon) {
        var formData = new FormData();
        formData.append('lat', lat);
        formData.append('lon', lon);

        // Clear previous results and show loading indicators
        if (imageOverlay) {
            map.removeLayer(imageOverlay);
            imageOverlay = null;
        }
        removePreviewRectangle();
        imageContainer.innerHTML = `
            <div class="text-center">
                <p><i class="fas fa-spinner fa-spin"></i> Loading image...</p>
            </div>`;
        predictionContainer.innerHTML = `
            <div class="text-center">
                <p><i class="fas fa-spinner fa-spin"></i> Loading Grad-CAM visualizations...</p>
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

            // Calculate square bounds for the image overlay
            const squareBounds = calculateSquareBounds(lat, lon);

            // Use the square bounds instead of data.bounds
            imageOverlay = L.imageOverlay(data.image_url, squareBounds).addTo(map);
            map.fitBounds(squareBounds);

            // Display the image in the image-container
            imageContainer.innerHTML = `
                <div class="mb-4">
                    <img src="${data.image_url}" alt="RGB Image Patch" class="img-fluid mb-3 rounded" style="width: 300px;">
                    <p class="card-text"><small class="text-muted">Multispectral TIFF saved as: ${data.tiff_file}</small></p>
                    <p class="card-text"><small class="text-muted">Coords: ${lat.toFixed(4)}, ${lon.toFixed(4)}</small></p>
                </div>`;

            // Step 2: Fetch predictions
            var experimentSelect = document.getElementById('experiment-select');
            var selectedExperiment = experimentSelect.value;

            if (!selectedExperiment) {
                imageContainer.innerHTML += `<div class="alert alert-warning mt-3">Please select an experiment before clicking on the map.</div>`;
                predictionContainer.innerHTML = `<div class="text-muted py-5"><i class="bi bi-bar-chart"></i> Grad-CAM visualizations will appear here</div>`;
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
            return response.json();
        })
        .then(data => {
            console.log('predict_from_map data:', data);
            if (data.error) {
                imageContainer.innerHTML += `<div class="alert alert-danger mt-3">Error: ${data.error}</div>`;
                predictionContainer.innerHTML = `<div class="text-muted py-5"><i class="bi bi-bar-chart"></i> Grad-CAM visualizations will appear here</div>`;
                return;
            }

            // Append predictions to the image-container
            let predictionHtml = `
                <div class="mt-4">
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
                </div>`;
            imageContainer.innerHTML += predictionHtml;

            // Render Grad-CAM visualizations in the prediction-container
            let gradcamHtml = `
                <div>
                    <h5 class="fw-bold">Grad-CAM Visualizations</h5>
                    <div class="row">`;
            for (let [className, url] of Object.entries(data.gradcam)) {
                gradcamHtml += `
                    <div class="col-md-6">
                        <img src="${url}" alt="Grad-CAM for ${className}" class="img-fluid rounded mb-2">
                        <p class="text-center">${className}</p>
                    </div>`;
            }
            gradcamHtml += `
                    </div>
                    <h5 class="fw-bold">Color-Coded Grad-CAM</h5>
                    <img src="${data.gradcam_colourcoded_.combined.url}" alt="Color-Coded Grad-CAM" class="img-fluid rounded mb-2" style="width: 300px;">
                    <div class="legend-title">Legend</div>
                    <div class="legend-container">`;
            
            for (let [category, color] of Object.entries(data.gradcam_colourcoded_.combined.legend)) {
                gradcamHtml += `
                    <div class="legend-item">
                        <div class="legend-color-box" style="background-color: ${color};"></div>
                        <span class="legend-text">${category}</span>
                    </div>`;
            }
            gradcamHtml += `
                    </div>
                </div>`;
            predictionContainer.innerHTML = gradcamHtml;
        })
        .catch(error => {
            console.error('Error during fetch process:', error);
            const errorMsg = `<div class="alert alert-danger mt-3">Error: ${error.message}</div>`;
            imageContainer.innerHTML = errorMsg;
            predictionContainer.innerHTML = `<div class="text-muted py-5"><i class="bi bi-bar-chart"></i> Grad-CAM visualizations will appear here</div>`;
        });
    }

    // Mousemove event to show the preview rectangle
    map.on('mousemove', function(e) {
        var lat = e.latlng.lat;
        var lon = e.latlng.lng;
        drawPreviewRectangle(lat, lon);
    });

    // Click event to fetch the patch
    map.on('click', function(e) {
        var lat = e.latlng.lat;
        var lon = e.latlng.lng;
        fetchImageAndPrediction(lat, lon);
    });

    // Remove the preview rectangle when the mouse leaves the map
    map.on('mouseout', function() {
        removePreviewRectangle();
    });
});