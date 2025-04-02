function renderBubbleChart(experimentsData) {
  // Extract arrays for the plot
  const xValues = experimentsData.map(d => d.training_time_min);
  const yValues = experimentsData.map(d => d.val_f2);
  const sizes = experimentsData.map(d => d.model_size_MB);
  const labels = experimentsData.map(d => d.model);
  const experimentNames = experimentsData.map(d => d.experiment);

  // Scale bubble sizes
  const minSize = 15, maxSize = 75;
  const minVal = Math.min(...sizes);
  const maxVal = Math.max(...sizes);
  const sizeValues = sizes.map(s => {
    const normalized = (s - minVal) / (maxVal - minVal + 1e-9);
    return minSize + (maxSize - minSize) * Math.sqrt(normalized); // Square root scaling
  });

  // Get unique models (not model types)
  const uniqueModels = [...new Set(experimentsData.map(d => d.model))];

  // Create a color scale for unique models
  const colorScale = Plotly.d3.scale.category20(); 
  const modelColors = {};
  uniqueModels.forEach((model, i) => {
    modelColors[model] = colorScale(i % 20); // Assign a unique color to each model
  });

  // Group data by individual model for separate traces
  const traces = [];
  uniqueModels.forEach(model => {
    const indices = experimentsData
      .map((d, i) => (d.model === model ? i : -1))
      .filter(i => i !== -1);

    const trace = {
      x: indices.map(i => xValues[i]),
      y: indices.map(i => yValues[i]),
      text: indices.map(i => labels[i]),
      hovertext: indices.map(i => experimentNames[i]),
      hovertemplate: 'Model: %{text}<br>Experiment: %{hovertext}<br>Training Time: %{x} minutes<br>F2 Score: %{y}<extra></extra>',
      mode: 'markers', // Labels on hover only
      name: model, 
      marker: {
        size: indices.map(i => sizeValues[i]),
        color: modelColors[model], 
        opacity: 0.9, 
        line: { width: 1, color: 'black' }
      },
      type: 'scatter'
    };
    traces.push(trace);
  });

  // Chart Layout with Zoom & Pan
  const layout = {
    title: 'F2 Score vs. Training Time',
    xaxis: { 
      title: 'Training Time (minutes)', 
      automargin: true, 
      autorange: true, 
      showgrid: false 
    },
    yaxis: { 
      title: 'F2 Score', 
      range: [0, 1], 
      automargin: true, 
      autorange: true, 
      showgrid: false 
    },
    hovermode: 'closest',
    dragmode: 'pan',
    scrollZoom: true,
    responsive: true,
    showlegend: true, // Show legend
    margin: { l: 50, r: 50, b: 50, t: 50 }, 
    grid: { rows: 1, columns: 1 }
  };

  // Render the Bubble Chart
  Plotly.newPlot('bubbleChart', traces, layout, { responsive: true });
}

// Toggle Select All / Deselect All Models
document.getElementById('toggleSelectAll').addEventListener('click', function() {
  let checkboxes = document.querySelectorAll('.model-checkbox');
  let allChecked = [...checkboxes].every(cb => cb.checked);
  checkboxes.forEach(cb => cb.checked = !allChecked);
});
