// bubble_chart.js
function renderBubbleChart(experimentsData) {
    // Extract arrays for the plot
    const xValues = experimentsData.map(d => d.training_time_min);
    const yValues = experimentsData.map(d => d.val_f2);
    const sizes = experimentsData.map(d => d.model_size_MB);
    const labels = experimentsData.map(d => d.model);
    const experimentNames = experimentsData.map(d => d.experiment); // Full experiment name for hover
  
    // Determine bubble colors based on the model type
    const colors = experimentsData.map(d => {
      let model = d.arch_type.toLowerCase();
      if (model.includes('custom_model')) return 'purple';
      else if (model.includes('resnet') || model.includes('densenet') || model.includes('vgg')) return 'blue';
      else if (model.includes('efficient')) return 'green';
      else if (model.includes('transformer') || model.includes('vit') || model.includes('swin')) return 'orange';
      else return 'gray';
    });
  
    // Scale bubble sizes
    const minSize = 10, maxSize = 60;
    const minVal = Math.min(...sizes);
    const maxVal = Math.max(...sizes);
    const sizeValues = sizes.map(s => ((s - minVal) / (maxVal - minVal + 1e-9)) * (maxSize - minSize) + minSize);
  
    // Bubble Chart Trace
    const trace = {
      x: xValues,
      y: yValues,
      text: labels,
      hovertext: experimentNames,
      hovertemplate: 'Experiment: %{hovertext}<br>Training Time: %{x} minutes<br>F2 Score: %{y}<extra></extra>',
      mode: 'markers+text',
      textposition: 'top center',
      marker: {
        size: sizeValues,
        color: colors,
        opacity: 0.7,
        line: { width: 1, color: 'black' }
      },
      type: 'scatter'
    };
  
    // Chart Layout with Zoom & Pan
    const layout = {
      title: 'F2 Score vs. Training Time',
      xaxis: { title: 'Training Time (minutes)', automargin: true, autorange: true },
      yaxis: { title: 'F2 Score', range: [0, 1], automargin: true, autorange: true },
      hovermode: 'closest',
      dragmode: 'pan',
      scrollZoom: true,
      responsive: true,
      showlegend: false,
      margin: { l: 50, r: 50, b: 50, t: 50 },
      grid: { rows: 1, columns: 1 }
    };
  
    // Render the Bubble Chart
    Plotly.newPlot('bubbleChart', [trace], layout, { responsive: true });
  }
  
  // Toggle Select All / Deselect All Models
  document.getElementById('toggleSelectAll').addEventListener('click', function() {
    let checkboxes = document.querySelectorAll('.model-checkbox');
    let allChecked = [...checkboxes].every(cb => cb.checked);
    checkboxes.forEach(cb => cb.checked = !allChecked);
  });