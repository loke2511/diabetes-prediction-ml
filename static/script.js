// Diabetes Prediction - Client-side JavaScript

// Initialize particles
function createParticles() {
    const particleCount = 20;
    const container = document.querySelector('.animated-bg');
    
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        
        const size = Math.random() * 100 + 50;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        particle.style.left = `${Math.random() * 100}%`;
        particle.style.top = `${Math.random() * 100}%`;
        particle.style.animationDelay = `${Math.random() * 20}s`;
        particle.style.animationDuration = `${Math.random() * 10 + 15}s`;
        
        const colors = ['#6366f1', '#ec4899', '#10b981', '#f59e0b'];
        particle.style.background = colors[Math.floor(Math.random() * colors.length)];
        
        container.appendChild(particle);
    }
}

// Update range input displays
function setupRangeInputs() {
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    
    rangeInputs.forEach(input => {
        const valueDisplay = document.getElementById(`${input.id}-value`);
        if (valueDisplay) {
            input.addEventListener('input', (e) => {
                valueDisplay.textContent = parseFloat(e.target.value).toFixed(1);
            });
        }
    });
}

// Validate form inputs
function validateForm() {
    const inputs = document.querySelectorAll('.form-input');
    let isValid = true;
    
    inputs.forEach(input => {
        if (input.value === '' || input.value < 0) {
            isValid = false;
            input.style.borderColor = 'var(--danger-color)';
        } else {
            input.style.borderColor = 'var(--border-color)';
        }
    });
    
    return isValid;
}

// Show alert message
function showAlert(message, type = 'error') {
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} show`;
    alertDiv.textContent = message;
    
    const form = document.getElementById('prediction-form');
    form.insertBefore(alertDiv, form.firstChild);
    
    setTimeout(() => {
        alertDiv.classList.remove('show');
        setTimeout(() => alertDiv.remove(), 300);
    }, 5000);
}

// Make prediction
async function makePrediction(event) {
    event.preventDefault();
    
    // Validate form
    if (!validateForm()) {
        showAlert('Please fill in all fields with valid values.');
        return;
    }
    
    // Get form data
    const formData = {
        pregnancies: parseFloat(document.getElementById('pregnancies').value),
        glucose: parseFloat(document.getElementById('glucose').value),
        bloodPressure: parseFloat(document.getElementById('blood-pressure').value),
        skinThickness: parseFloat(document.getElementById('skin-thickness').value),
        insulin: parseFloat(document.getElementById('insulin').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        diabetesPedigreeFunction: parseFloat(document.getElementById('dpf').value),
        age: parseFloat(document.getElementById('age').value)
    };
    
    // Show loading
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    
    loadingDiv.classList.add('show');
    resultsDiv.classList.remove('show');
    
    // Scroll to loading
    loadingDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(formData)
        });
        
        const data = await response.json();
        
        if (!response.ok || !data.success) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showAlert(`Error: ${error.message}`);
        loadingDiv.classList.remove('show');
    }
}

// Display prediction results
function displayResults(data) {
    const loadingDiv = document.getElementById('loading');
    const resultsDiv = document.getElementById('results');
    
    // Hide loading
    loadingDiv.classList.remove('show');
    
    // Update result status
    const resultIcon = document.getElementById('result-icon');
    const resultStatus = document.getElementById('result-status');
    const resultProbability = document.getElementById('result-probability');
    const resultRisk = document.getElementById('result-risk');
    
    if (data.prediction === 1) {
        resultIcon.textContent = '⚠️';
        resultStatus.textContent = 'Diabetes Risk Detected';
        resultStatus.style.color = 'var(--danger-color)';
    } else {
        resultIcon.textContent = '✅';
        resultStatus.textContent = 'Low Diabetes Risk';
        resultStatus.style.color = 'var(--success-color)';
    }
    
    resultProbability.textContent = `${data.probability.diabetes}%`;
    resultRisk.textContent = `${data.risk_level} Risk`;
    resultRisk.className = `result-risk risk-${data.risk_level.toLowerCase()}`;
    
    // Update probability bars
    updateProbabilityBars(data.probability);
    
    // Display recommendations
    displayRecommendations(data.recommendations);
    
    // Update model info
    document.getElementById('model-name').textContent = data.model_used;
    document.getElementById('prediction-time').textContent = data.timestamp;
    
    // Show results with animation
    resultsDiv.classList.add('show');
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Update probability bars
function updateProbabilityBars(probability) {
    const noDiabetesBar = document.getElementById('probability-no-diabetes');
    const diabetesBar = document.getElementById('probability-diabetes');
    
    // Reset widths
    noDiabetesBar.style.width = '0%';
    diabetesBar.style.width = '0%';
    
    // Animate bars
    setTimeout(() => {
        noDiabetesBar.style.width = `${probability.no_diabetes}%`;
        diabetesBar.style.width = `${probability.diabetes}%`;
    }, 100);
}

// Display recommendations
function displayRecommendations(recommendations) {
    const container = document.getElementById('recommendations-list');
    container.innerHTML = '';
    
    recommendations.forEach((rec, index) => {
        const recDiv = document.createElement('div');
        recDiv.className = `recommendation priority-${rec.priority}`;
        recDiv.style.animationDelay = `${index * 0.1}s`;
        
        recDiv.innerHTML = `
            <div class="recommendation-icon">${rec.icon}</div>
            <div class="recommendation-content">
                <div class="recommendation-category">${rec.category}</div>
                <div class="recommendation-message">${rec.message}</div>
            </div>
        `;
        
        container.appendChild(recDiv);
    });
}

// Reset form
function resetForm() {
    document.getElementById('prediction-form').reset();
    document.getElementById('results').classList.remove('show');
    
    // Reset range value displays
    const rangeInputs = document.querySelectorAll('input[type="range"]');
    rangeInputs.forEach(input => {
        const valueDisplay = document.getElementById(`${input.id}-value`);
        if (valueDisplay) {
            valueDisplay.textContent = parseFloat(input.value).toFixed(1);
        }
    });
    
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Load sample data for testing
function loadSampleData() {
    // Sample data for a person with moderate diabetes risk
    const sampleData = {
        pregnancies: 2,
        glucose: 140,
        bloodPressure: 85,
        skinThickness: 25,
        insulin: 120,
        bmi: 32.5,
        dpf: 0.8,
        age: 45
    };
    
    Object.keys(sampleData).forEach(key => {
        const input = document.getElementById(key);
        if (input) {
            input.value = sampleData[key];
            
            // Update range display if applicable
            const valueDisplay = document.getElementById(`${key}-value`);
            if (valueDisplay) {
                valueDisplay.textContent = parseFloat(sampleData[key]).toFixed(1);
            }
        }
    });
    
    showAlert('Sample data loaded! Click "Predict Diabetes Risk" to see results.', 'success');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    createParticles();
    setupRangeInputs();
    
    // Setup form submission
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', makePrediction);
    }
    
    // Setup reset button
    const resetBtn = document.getElementById('reset-btn');
    if (resetBtn) {
        resetBtn.addEventListener('click', resetForm);
    }
    
    // Setup sample data button
    const sampleBtn = document.getElementById('sample-btn');
    if (sampleBtn) {
        sampleBtn.addEventListener('click', loadSampleData);
    }
    
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth' });
            }
        });
    });
    
    // Add input validation on blur
    document.querySelectorAll('.form-input').forEach(input => {
        input.addEventListener('blur', () => {
            if (input.value !== '' && input.value >= 0) {
                input.style.borderColor = 'var(--success-color)';
            }
        });
    });
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to submit form
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        const form = document.getElementById('prediction-form');
        if (form) {
            form.dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to reset
    if (e.key === 'Escape') {
        resetForm();
    }
});
