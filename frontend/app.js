// Global state
let eventSource = null;
let currentSteps = [];
let completedSteps = 0;
let totalSteps = 0;

// DOM elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const userTextInput = document.getElementById('userText');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultSection = document.getElementById('resultSection');
const progressSection = document.getElementById('progressSection');
const toolsSection = document.getElementById('toolsSection');
const resultCard = document.getElementById('resultCard');
const probabilityCircle = document.getElementById('probabilityCircle');
const probabilityValue = document.getElementById('probabilityValue');
const probabilityLabel = document.getElementById('probabilityLabel');
const explanationText = document.getElementById('explanationText');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const currentStep = document.getElementById('currentStep');
const toolsGrid = document.getElementById('toolsGrid');

// Initialize tool cards
const toolNames = {
    'ReverseImageSearch': 'Reverse Image Search',
    'SynthIDDetection': 'SynthID Detection',
    'VisualForensicsAgent': 'Visual Forensics',
    'JudgeSystem': 'Judge System Debate',
    'AIMetadataAnalyzer': 'AI Metadata Analysis'
};

// Setup upload functionality
uploadBox.addEventListener('click', () => imageInput.click());
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#6366f1';
});
uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '';
});
uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleFileSelect();
    }
});

imageInput.addEventListener('change', handleFileSelect);

function handleFileSelect() {
    if (imageInput.files.length > 0) {
        analyzeBtn.disabled = false;
        uploadBox.querySelector('.upload-content p').textContent = `Selected: ${imageInput.files[0].name}`;
    }
}

// Analyze button handler
analyzeBtn.addEventListener('click', async () => {
    if (!imageInput.files.length) return;

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = 'Analyzing...';

    // Reset UI
    resetUI();

    // Show sections
    resultSection.style.display = 'block';
    progressSection.style.display = 'block';
    toolsSection.style.display = 'block';

    // Create form data
    const formData = new FormData();
    formData.append('image', imageInput.files[0]);
    formData.append('text', userTextInput.value || '');

    try {
        // Start analysis
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to start analysis');
        }

        // Connect to SSE stream
        connectToProgressStream();

    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = 'Analyze Image';
    }
});

function resetUI() {
    completedSteps = 0;
    totalSteps = 0;
    currentSteps = [];
    toolsGrid.innerHTML = '';

    // Initialize tool cards
    Object.keys(toolNames).forEach(toolName => {
        createToolCard(toolName, toolNames[toolName]);
    });

    // Reset result display
    probabilityValue.textContent = '--';
    probabilityLabel.textContent = 'Deepfake Probability';
    explanationText.textContent = 'Analysis in progress...';
    progressFill.style.width = '0%';
    progressText.textContent = '0%';
    currentStep.textContent = 'Waiting to start...';

    // Reset probability circle
    updateProbabilityCircle(null);
}

function createToolCard(toolName, displayName) {
    const card = document.createElement('div');
    card.className = 'tool-card pending';
    card.id = `tool-${toolName}`;
    card.innerHTML = `
        <div class="tool-header">
            <span class="tool-name">${displayName}</span>
            <span class="tool-status pending">Pending</span>
        </div>
        <div class="tool-content" id="tool-content-${toolName}">Waiting to start...</div>
    `;
    toolsGrid.appendChild(card);
}

function updateToolCard(toolName, status, content) {
    const card = document.getElementById(`tool-${toolName}`);
    const statusEl = card.querySelector('.tool-status');
    const contentEl = document.getElementById(`tool-content-${toolName}`);

    card.className = `tool-card ${status}`;
    statusEl.className = `tool-status ${status}`;
    statusEl.textContent = status.charAt(0).toUpperCase() + status.slice(1);

    if (content) {
        contentEl.textContent = formatToolContent(content);
    }
}

function formatToolContent(content) {
    if (typeof content === 'string') {
        return content;
    }
    if (typeof content === 'object') {
        try {
            return JSON.stringify(content, null, 2);
        } catch (e) {
            return String(content);
        }
    }
    return String(content);
}

function connectToProgressStream() {
    if (eventSource) {
        eventSource.close();
    }

    eventSource = new EventSource('/api/progress');

    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleProgressUpdate(data);
        } catch (error) {
            console.error('Error parsing progress update:', error);
        }
    };

    eventSource.onerror = (error) => {
        console.error('SSE error:', error);
        // Don't close on error, let it reconnect
    };
}

function handleProgressUpdate(update) {
    const { type, data } = update;

    switch (type) {
        case 'start':
            totalSteps = data.total_steps || 0;
            currentStep.textContent = 'Pipeline started';
            break;

        case 'step_start':
            updateToolCard(data.step, 'running', 'Running analysis...');
            currentStep.textContent = `Running: ${data.display_name}...`;
            break;

        case 'step_complete':
            updateToolCard(data.step, 'complete', data.result?.content);
            completedSteps++;
            updateProgress();
            break;

        case 'step_error':
            updateToolCard(data.step, 'error', `Error: ${data.error}`);
            completedSteps++;
            updateProgress();
            break;

        case 'final_analysis_start':
            currentStep.textContent = 'Performing final analysis...';
            break;

        case 'final_result':
            displayFinalResult(data);
            break;

        case 'complete':
            currentStep.textContent = 'Analysis completed!';
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Image';
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            break;

        case 'error':
            showError(data.error);
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Image';
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            break;
    }
}

function updateProgress() {
    if (totalSteps > 0) {
        const percentage = Math.round((completedSteps / totalSteps) * 100);
        progressFill.style.width = `${percentage}%`;
        progressText.textContent = `${percentage}%`;
    }
}

function displayFinalResult(data) {
    const probability = data.probability_score;
    const explanation = data.explanation || 'No explanation provided.';

    // Update probability display
    if (probability !== null && probability !== undefined) {
        probabilityValue.textContent = probability;
        updateProbabilityCircle(probability);

        // Update label based on probability
        if (probability >= 70) {
            probabilityLabel.textContent = 'High Deepfake Probability';
        } else if (probability >= 40) {
            probabilityLabel.textContent = 'Moderate Deepfake Probability';
        } else {
            probabilityLabel.textContent = 'Low Deepfake Probability';
        }
    } else {
        probabilityValue.textContent = 'N/A';
        probabilityLabel.textContent = 'Unable to determine';
    }

    // Update explanation
    explanationText.textContent = explanation;

    // Update progress to 100%
    progressFill.style.width = '100%';
    progressText.textContent = '100%';
    currentStep.textContent = 'Final analysis complete!';
}

function updateProbabilityCircle(probability) {
    if (probability === null || probability === undefined) {
        probabilityCircle.style.background = 'conic-gradient(from 0deg, #334155 0%, #334155 100%)';
        return;
    }

    // Calculate gradient based on probability
    // 0-30: green (real), 30-70: yellow (uncertain), 70-100: red (fake)
    let color1, color2, color3;

    if (probability < 30) {
        color1 = '#10b981'; // green
        color2 = '#10b981';
        color3 = '#10b981';
    } else if (probability < 70) {
        color1 = '#f59e0b'; // yellow
        color2 = '#f59e0b';
        color3 = '#f59e0b';
    } else {
        color1 = '#ef4444'; // red
        color2 = '#ef4444';
        color3 = '#ef4444';
    }

    // Create gradient based on percentage
    const percentage = probability;
    const angle = (percentage / 100) * 360;

    probabilityCircle.style.background = `conic-gradient(from 0deg, ${color1} 0%, ${color2} ${angle}deg, #334155 ${angle}deg, #334155 100%)`;
}

function showError(message) {
    currentStep.textContent = `Error: ${message}`;
    currentStep.style.color = '#ef4444';
}

