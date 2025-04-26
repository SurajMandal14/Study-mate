/**
 * Utility Functions
 * Helper functions used across the application
 */

// Format date to readable string (e.g., "Mon, Jan 1, 2023")
function formatDate(dateString) {
  const date = new Date(dateString);
  const options = { 
    weekday: 'short', 
    year: 'numeric', 
    month: 'short', 
    day: 'numeric' 
  };
  
  return date.toLocaleDateString('en-US', options);
}

// Calculate days until a target date
function getDaysUntil(targetDateString) {
  const today = new Date();
  today.setHours(0, 0, 0, 0); // Reset time to start of day
  
  const targetDate = new Date(targetDateString);
  targetDate.setHours(0, 0, 0, 0);
  
  const diffTime = targetDate - today;
  const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
  
  return diffDays;
}

// Get appropriate icon for a subject
function getSubjectIcon(subjectName) {
  const subjectName_lower = subjectName.toLowerCase();
  
  if (subjectName_lower.includes('math') || subjectName_lower.includes('calculus') || subjectName_lower.includes('algebra')) {
    return 'fas fa-square-root-alt';
  } else if (subjectName_lower.includes('physics') || subjectName_lower.includes('mechanics')) {
    return 'fas fa-atom';
  } else if (subjectName_lower.includes('chemistry')) {
    return 'fas fa-flask';
  } else if (subjectName_lower.includes('biology') || subjectName_lower.includes('ecology')) {
    return 'fas fa-dna';
  } else if (subjectName_lower.includes('history') || subjectName_lower.includes('civilization')) {
    return 'fas fa-book';
  } else if (subjectName_lower.includes('geography') || subjectName_lower.includes('geology')) {
    return 'fas fa-globe-americas';
  } else if (subjectName_lower.includes('computer') || subjectName_lower.includes('programming')) {
    return 'fas fa-laptop-code';
  } else if (subjectName_lower.includes('language') || subjectName_lower.includes('english') || subjectName_lower.includes('literature')) {
    return 'fas fa-language';
  } else if (subjectName_lower.includes('art') || subjectName_lower.includes('painting')) {
    return 'fas fa-palette';
  } else if (subjectName_lower.includes('music')) {
    return 'fas fa-music';
  } else if (subjectName_lower.includes('psychology') || subjectName_lower.includes('sociology')) {
    return 'fas fa-brain';
  } else if (subjectName_lower.includes('economics') || subjectName_lower.includes('business')) {
    return 'fas fa-chart-line';
  } else if (subjectName_lower.includes('politics') || subjectName_lower.includes('government')) {
    return 'fas fa-landmark';
  } else if (subjectName_lower.includes('philosophy') || subjectName_lower.includes('ethics')) {
    return 'fas fa-balance-scale';
  } else {
    return 'fas fa-book-open'; // Default icon
  }
}

// Get the status of a day in the study plan (completed, today, overdue, upcoming)
function getDayStatus(date, dayPlan) {
  // If explicitly marked as completed
  if (dayPlan.completed) {
    return 'completed';
  }
  
  const today = new Date();
  today.setHours(0, 0, 0, 0); // Reset time to start of day
  
  const planDate = new Date(date);
  planDate.setHours(0, 0, 0, 0);
  
  // If this is today's plan
  if (planDate.getTime() === today.getTime()) {
    return 'today';
  }
  
  // If this plan date is in the past and not completed
  if (planDate < today) {
    return 'overdue';
  }
  
  // Otherwise it's an upcoming plan
  return 'upcoming';
}

// Generate a unique ID (useful for various elements)
function generateUniqueId() {
  return 'id_' + Math.random().toString(36).substr(2, 9);
}

// Truncate text to a certain length with ellipsis
function truncateText(text, maxLength = 100) {
  if (!text || text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}

// Validate email format
function isValidEmail(email) {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

// Check if a string is a valid date
function isValidDate(dateString) {
  const date = new Date(dateString);
  return !isNaN(date.getTime());
}

// Format time in minutes to hours and minutes (e.g., "2h 30m")
function formatTime(minutes) {
  if (!minutes) return '0m';
  
  const hours = Math.floor(minutes / 60);
  const mins = minutes % 60;
  
  if (hours === 0) return `${mins}m`;
  if (mins === 0) return `${hours}h`;
  
  return `${hours}h ${mins}m`;
}

// Calculate total study time for a day plan
function calculateDayStudyTime(dayPlan) {
  if (!dayPlan || !dayPlan.subjects) return 0;
  
  return dayPlan.subjects.reduce((total, subject) => {
    return total + (subject.duration_minutes || 0);
  }, 0);
}

// Calculate total study time for the entire plan
function calculateTotalStudyTime(plan) {
  if (!plan || !plan.daily_plans) return 0;
  
  return Object.values(plan.daily_plans).reduce((total, day) => {
    return total + calculateDayStudyTime(day);
  }, 0);
}

// Get a random motivational quote for studying
function getRandomMotivationalQuote() {
  const quotes = [
    { text: "The secret of getting ahead is getting started.", author: "Mark Twain" },
    { text: "Don't watch the clock; do what it does. Keep going.", author: "Sam Levenson" },
    { text: "Success is no accident. It is hard work, perseverance, learning, studying, sacrifice and most of all, love of what you are doing.", author: "Pele" },
    { text: "The beautiful thing about learning is that no one can take it away from you.", author: "B.B. King" },
    { text: "The more that you read, the more things you will know. The more that you learn, the more places you'll go.", author: "Dr. Seuss" },
    { text: "The expert in anything was once a beginner.", author: "Helen Hayes" },
    { text: "There are no shortcuts to any place worth going.", author: "Beverly Sills" },
    { text: "Believe you can and you're halfway there.", author: "Theodore Roosevelt" },
    { text: "The difference between try and triumph is just a little umph!", author: "Marvin Phillips" },
    { text: "The future belongs to those who believe in the beauty of their dreams.", author: "Eleanor Roosevelt" }
  ];
  
  const randomIndex = Math.floor(Math.random() * quotes.length);
  return quotes[randomIndex];
}

// Convert array of subjects to a format suitable for the API
function formatSubjectsForAPI(subjects) {
  return subjects.map(subject => ({
    name: subject.name,
    topics: subject.topics,
    confidence: subject.confidence,
    weight: subject.weight || 3
  }));
}

// Export functions to global scope for HTML access
window.formatDate = formatDate;
window.getDaysUntil = getDaysUntil;
window.getSubjectIcon = getSubjectIcon;
window.getDayStatus = getDayStatus;
window.generateUniqueId = generateUniqueId;
window.truncateText = truncateText;
window.isValidEmail = isValidEmail;
window.isValidDate = isValidDate;
window.formatTime = formatTime;
window.calculateDayStudyTime = calculateDayStudyTime;
window.calculateTotalStudyTime = calculateTotalStudyTime;
window.getRandomMotivationalQuote = getRandomMotivationalQuote;
window.formatSubjectsForAPI = formatSubjectsForAPI; 