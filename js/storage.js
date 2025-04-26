/**
 * Storage Module
 * Handles saving and loading data from localStorage
 */

// Storage keys
const STORAGE_KEYS = {
  SUBJECTS: 'lastMinRev_subjects',
  MANUAL_SUBJECTS: 'lastMinRev_manualSubjects',
  STUDY_PLAN: 'lastMinRev_studyPlan',
  PROGRESS: 'lastMinRev_progress',
  SETTINGS: 'lastMinRev_settings',
  USER_INFO: 'lastMinRev_userInfo'
};

// Save extracted subjects to localStorage
function saveExtractedSubjects(subjects) {
  localStorage.setItem(STORAGE_KEYS.SUBJECTS, JSON.stringify(subjects));
}

// Load extracted subjects from localStorage
function loadExtractedSubjects() {
  const subjects = localStorage.getItem(STORAGE_KEYS.SUBJECTS);
  return subjects ? JSON.parse(subjects) : [];
}

// Save manually added subjects
function saveManualSubjects(subjects) {
  localStorage.setItem(STORAGE_KEYS.MANUAL_SUBJECTS, JSON.stringify(subjects));
}

// Load manually added subjects
function loadManualSubjects() {
  const subjects = localStorage.getItem(STORAGE_KEYS.MANUAL_SUBJECTS);
  return subjects ? JSON.parse(subjects) : [];
}

// Save the complete study plan
function saveStudyPlan(plan) {
  localStorage.setItem(STORAGE_KEYS.STUDY_PLAN, JSON.stringify(plan));
}

// Load the study plan
function loadStudyPlan() {
  const plan = localStorage.getItem(STORAGE_KEYS.STUDY_PLAN);
  return plan ? JSON.parse(plan) : null;
}

// Save progress tracking data
function saveProgress(progress) {
  localStorage.setItem(STORAGE_KEYS.PROGRESS, JSON.stringify(progress));
}

// Load progress tracking data
function loadProgress() {
  const progress = localStorage.getItem(STORAGE_KEYS.PROGRESS);
  return progress ? JSON.parse(progress) : {
    completedDays: [],
    lastUpdated: new Date().toISOString()
  };
}

// Save user settings
function saveSettings(settings) {
  localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
}

// Load user settings
function loadSettings() {
  const settings = localStorage.getItem(STORAGE_KEYS.SETTINGS);
  return settings ? JSON.parse(settings) : {
    theme: 'light',
    reminderTime: '19:00',
    enableNotifications: true,
    studySessionLength: 45, // minutes
    breakLength: 15, // minutes
    enableSound: true
  };
}

// Save user information
function saveUserInfo(userInfo) {
  localStorage.setItem(STORAGE_KEYS.USER_INFO, JSON.stringify(userInfo));
}

// Load user information
function loadUserInfo() {
  const userInfo = localStorage.getItem(STORAGE_KEYS.USER_INFO);
  return userInfo ? JSON.parse(userInfo) : {
    name: '',
    email: '',
    examDate: '',
    examName: ''
  };
}

// Clear all stored data (for reset functionality)
function clearAllData() {
  localStorage.removeItem(STORAGE_KEYS.SUBJECTS);
  localStorage.removeItem(STORAGE_KEYS.MANUAL_SUBJECTS);
  localStorage.removeItem(STORAGE_KEYS.STUDY_PLAN);
  localStorage.removeItem(STORAGE_KEYS.PROGRESS);
  localStorage.removeItem(STORAGE_KEYS.SETTINGS);
  localStorage.removeItem(STORAGE_KEYS.USER_INFO);
}

// Export to a JSON file for backup
function exportDataToFile() {
  const data = {
    extractedSubjects: loadExtractedSubjects(),
    manualSubjects: loadManualSubjects(),
    studyPlan: loadStudyPlan(),
    progress: loadProgress(),
    settings: loadSettings(),
    userInfo: loadUserInfo(),
    exportDate: new Date().toISOString()
  };
  
  const dataStr = JSON.stringify(data, null, 2);
  const dataBlob = new Blob([dataStr], { type: 'application/json' });
  const url = URL.createObjectURL(dataBlob);
  
  const link = document.createElement('a');
  link.href = url;
  const fileName = `lastMinRev_backup_${new Date().toISOString().slice(0, 10)}.json`;
  link.download = fileName;
  link.click();
  
  URL.revokeObjectURL(url);
  
  return fileName;
}

// Import data from a JSON file
function importDataFromJSON(jsonData) {
  try {
    const data = JSON.parse(jsonData);
    
    // Validate the imported data
    if (!data.extractedSubjects || !data.manualSubjects || !data.studyPlan) {
      throw new Error('Invalid backup file format');
    }
    
    // Import the data
    if (data.extractedSubjects) saveExtractedSubjects(data.extractedSubjects);
    if (data.manualSubjects) saveManualSubjects(data.manualSubjects);
    if (data.studyPlan) saveStudyPlan(data.studyPlan);
    if (data.progress) saveProgress(data.progress);
    if (data.settings) saveSettings(data.settings);
    if (data.userInfo) saveUserInfo(data.userInfo);
    
    return {
      success: true,
      message: 'Data imported successfully'
    };
  } catch (error) {
    return {
      success: false,
      message: `Import failed: ${error.message}`
    };
  }
}

// Check if there is saved data available
function hasSavedData() {
  return (
    localStorage.getItem(STORAGE_KEYS.SUBJECTS) !== null ||
    localStorage.getItem(STORAGE_KEYS.MANUAL_SUBJECTS) !== null ||
    localStorage.getItem(STORAGE_KEYS.STUDY_PLAN) !== null
  );
}

// Export functions to global scope for HTML access
window.saveExtractedSubjects = saveExtractedSubjects;
window.loadExtractedSubjects = loadExtractedSubjects;
window.saveManualSubjects = saveManualSubjects;
window.loadManualSubjects = loadManualSubjects;
window.saveStudyPlan = saveStudyPlan;
window.loadStudyPlan = loadStudyPlan;
window.saveProgress = saveProgress;
window.loadProgress = loadProgress;
window.saveSettings = saveSettings;
window.loadSettings = loadSettings;
window.saveUserInfo = saveUserInfo;
window.loadUserInfo = loadUserInfo;
window.clearAllData = clearAllData;
window.exportDataToFile = exportDataToFile;
window.importDataFromJSON = importDataFromJSON;
window.hasSavedData = hasSavedData; 