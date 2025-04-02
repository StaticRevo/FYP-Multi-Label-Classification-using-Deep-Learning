// map.js
var map = L.map('map').setView([35.9, 14.5], 11); // Default: Malta

// Add Esri World Imagery (satellite view)
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    maxZoom: 19,
    attribution: 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
}).addTo(map);

var imageOverlay = null;

function fetchImage(lat, lon) {
    var formData = new FormData();
    formData.append('lat', lat);
    formData.append('lon', lon);

    document.getElementById('image-container').innerHTML = 'Loading...';

    fetch('/get_image', {
        method: 'POST',
        body: formData,
        headers: { 'Accept': 'application/json' }
    })
    .then(response => response.json())
    .then(data => {
        if (imageOverlay) {
            map.removeLayer(imageOverlay);
        }
        imageOverlay = L.imageOverlay(data.image_url, data.bounds).addTo(map);
        map.fitBounds(data.bounds);
        document.getElementById('image-container').innerHTML = `
            <img src="${data.image_url}" alt="RGB Image" style="max-width: 300px;">
            <p>Multispectral TIFF saved as: ${data.tiff_file}</p>
        `;
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('image-container').innerHTML = 'Error fetching image.';
    });
}

map.on('click', function(e) {
    var lat = e.latlng.lat;
    var lon = e.latlng.lng;
    fetchImage(lat, lon);
});