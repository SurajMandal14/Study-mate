/**
 * StudyMate - Smart Study Planner
 * Main application logic
 */

// Global variables
let darkMode = false;
let focusMode = false;
let grokApiKey = '';
let grokApiAuthenticated = false;
let extractedSubjects = [];
let manualSubjects = [
  { name: 'Mathematics', topics: ['Algebra', 'Calculus', 'Geometry'], weight: 5 },
  { name: 'Physics', topics: ['Mechanics', 'Optics', 'Electromagnetism'], weight: 3 }
];
let studyPlan = {};
let progress = {};
let chart;

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
  initializeApp();
});

function initializeApp() {
  // Show API authentication message
  document.getElementById('grokApiAuthMessage').classList.remove('hidden');
  
  // Initially disable PDF upload until API is authenticated
  document.getElementById('pdfInput').disabled = true;
  document.getElementById('pdfInput').classList.add('opacity-50', 'cursor-not-allowed');
  document.getElementById('apiErrorMsg').classList.remove('hidden');
  document.getElementById('apiErrorMsg').textContent = 'Please verify your Grok API key to enable AI-powered PDF analysis';
  
  // Set up event listeners
  setupEventListeners();
  
  // Render initial state for subject weights
  renderSubjectWeights();
  
  // Show random motivational quote
  showRandomQuote();
}

function setupEventListeners() {
  // Dark mode toggle
  document.getElementById('darkModeBtn').addEventListener('click', () => {
    toggleDarkMode(!darkMode);
  });
  
  // Focus mode toggle
  document.getElementById('focusModeBtn').onclick = function() {
    focusMode = !focusMode;
    toggleFocusMode();
  };
  
  // API key save
  document.getElementById('grokSaveBtn').onclick = verifyApiKey;
  
  // PDF upload
  document.getElementById('pdfInput').addEventListener('change', handlePdfUpload);
  
  // Manual entry toggle
  document.getElementById('toggleManualAdd').onclick = function() {
    const el = document.getElementById('manualAddSection');
    el.classList.toggle('hidden');
    if (!el.classList.contains('hidden')) renderManualSubjects();
  };
  
  // Add subject button
  document.getElementById('addSubjectBtn').onclick = function() {
    addNewManualSubject();
  };
  
  // Plan generation form
  document.getElementById('planGenForm').onsubmit = generateStudyPlan;
  
  // Routine preference dropdown
  document.getElementById('routinePref').onchange = function() {
    document.getElementById('customRoutineInputs').classList.toggle('hidden', this.value !== 'custom');
  };
  
  // Export PDF button
  document.getElementById('exportPdfBtn').addEventListener('click', exportToPdf);
}

function toggleDarkMode(enable) {
  darkMode = enable;
  document.body.classList.toggle('darkmode', darkMode);
  
  // Update all cards and inputs with dark mode classes
  if (darkMode) {
    document.querySelectorAll('.bg-white').forEach(el => {
      el.classList.remove('bg-white');
      el.classList.add('dark-card');
    });
    
    document.querySelectorAll('input, select, textarea').forEach(el => {
      el.classList.add('dark-input');
      // Add dark mode Tailwind classes
      el.classList.add('dark:bg-gray-700');
      el.classList.add('dark:text-gray-100');
      el.classList.add('dark:border-gray-600');
      // Remove light mode classes if present
      el.classList.remove('bg-gray-100');
      el.classList.add('bg-gray-700');
    });
    
    document.querySelectorAll('.border, [class*="border-"]').forEach(el => {
      el.classList.add('dark-border');
    });
    
    // Update light-specific background colors
    document.querySelectorAll('.bg-gray-50').forEach(el => {
      el.classList.remove('bg-gray-50');
      el.classList.add('dark:bg-gray-800');
    });
    
    document.getElementById('darkModeBtn').innerHTML = '<i class="fas fa-sun"></i> Light Mode';
  } else {
    document.querySelectorAll('.dark-card').forEach(el => {
      el.classList.add('bg-white');
      el.classList.remove('dark-card');
    });
    
    document.querySelectorAll('input, select, textarea').forEach(el => {
      el.classList.remove('dark-input');
      // Remove dark mode Tailwind classes for inputs
      el.classList.remove('dark:bg-gray-700');
      el.classList.remove('dark:text-gray-100');
      el.classList.remove('dark:border-gray-600');
      // Add light mode classes back
      el.classList.add('bg-gray-100');
      el.classList.remove('bg-gray-700');
    });
    
    document.querySelectorAll('.dark-border').forEach(el => {
      el.classList.remove('dark-border');
    });
    
    // Restore light-specific background colors
    document.querySelectorAll('.dark\\:bg-gray-800').forEach(el => {
      if (!el.classList.contains('bg-gray-50')) {
        el.classList.add('bg-gray-50');
      }
    });
    
    document.getElementById('darkModeBtn').innerHTML = '<i class="fas fa-moon"></i> Dark Mode';
  }
  
  // Update chart colors if chart exists
  if (chart) {
    chart.options.scales.x.grid.color = darkMode ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.05)';
    chart.update();
  }
  
  // Force refresh mermaid diagrams
  if (typeof mermaid !== 'undefined') {
    // Clear any existing mermaid diagrams and recreate them
    const mermaidDivs = document.querySelectorAll('.mermaid');
    if (mermaidDivs.length > 0) {
      const mermaidContents = [];
      mermaidDivs.forEach(div => {
        mermaidContents.push(div.textContent);
        div.removeAttribute('data-processed');
      });
      
      // Configure mermaid with new theme
      configureMermaid();
      
      // Rerun mermaid on the diagrams
      try {
        mermaid.run();
      } catch (e) {
        console.error('Error refreshing mermaid diagrams:', e);
        // If error, try recreating the diagrams
        mermaidDivs.forEach((div, i) => {
          div.textContent = mermaidContents[i];
        });
        setTimeout(() => mermaid.run(), 100);
      }
    }
  }
}

// Update UI when API is authenticated
function updateApiAuthUI(authenticated) {
  document.getElementById('grokApiAuthMessage').classList.toggle('hidden', authenticated);
  document.getElementById('pdfInput').disabled = !authenticated;
  document.getElementById('pdfInput').classList.toggle('opacity-50', !authenticated);
  document.getElementById('pdfInput').classList.toggle('cursor-not-allowed', !authenticated);
  
  if (authenticated) {
    document.getElementById('apiErrorMsg').classList.add('hidden');
  } else {
    document.getElementById('apiErrorMsg').classList.remove('hidden');
    document.getElementById('apiErrorMsg').textContent = 'Please verify your Grok API key to enable AI-powered PDF analysis';
  }
}

// Show PDF processing overlay
function showPdfProcessing(show) {
  document.getElementById('pdfProcessingOverlay').classList.toggle('hidden', !show);
}

// Show random motivational quote
function showRandomQuote() {
  const quotes = [
    "The North Remembers! Stay Strong.",
    "One topic at a time builds champions.",
    "Tiny progress is still progress — keep going.",
    "Every expert was once a beginner.",
    "Your efforts today, your success tomorrow!",
    "Difficult roads lead to beautiful destinations."
  ];
  document.getElementById('motivationQuote').textContent = quotes[Math.floor(Math.random()*quotes.length)];
}

// Format date to more readable format
function formatDate(dateString) {
  const date = new Date(dateString);
  const options = { month: 'short', day: 'numeric', year: 'numeric' };
  return date.toLocaleDateString('en-US', options);
}

// Get day status (today, past, upcoming)
function getDayStatus(dateString) {
  const today = new Date().toISOString().slice(0,10);
  const date = new Date(dateString);
  const nowDate = new Date(today);
  
  if (dateString === today) {
    return '<span class="px-2 py-0.5 bg-green-200 text-green-800 rounded-full">Today</span>';
  } else if (date < nowDate) {
    return '<span class="px-2 py-0.5 bg-gray-200 text-gray-800 rounded-full">Past</span>';
  } else {
    return '<span class="px-2 py-0.5 bg-blue-200 text-blue-800 rounded-full">Upcoming</span>';
  }
}

// Get appropriate icon for subject
function getSubjectIcon(subjectName) {
  const name = subjectName.toLowerCase();
  if (name.includes('math')) return 'fa-square-root-alt';
  if (name.includes('physics')) return 'fa-atom';
  if (name.includes('chem')) return 'fa-flask';
  if (name.includes('bio')) return 'fa-dna';
  if (name.includes('english') || name.includes('lit')) return 'fa-book';
  if (name.includes('history')) return 'fa-landmark';
  if (name.includes('geo')) return 'fa-globe-americas';
  if (name.includes('comp') || name.includes('cs')) return 'fa-laptop-code';
  // Default icon
  return 'fa-book';
}

// Export main functions to global scope for HTML access
window.toggleDarkMode = toggleDarkMode;
window.updateApiAuthUI = updateApiAuthUI;
window.showPdfProcessing = showPdfProcessing;
window.formatDate = formatDate;
window.getDayStatus = getDayStatus;
window.getSubjectIcon = getSubjectIcon; 