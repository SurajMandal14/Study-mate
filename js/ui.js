/**
 * UI Module
 * Handles user interface operations like rendering, animations, and UI interactions
 */

// Store UI references for commonly used elements
const UI = {
  // Main sections
  sections: {
    welcome: document.getElementById('welcomeSection'),
    subjects: document.getElementById('subjectsSection'),
    plan: document.getElementById('studyPlanSection'),
    progress: document.getElementById('progressSection'),
    settings: document.getElementById('settingsSection')
  },
  
  // Loaders
  loaders: {
    main: document.getElementById('mainLoader'),
    plan: document.getElementById('planLoader'),
    subjects: document.getElementById('subjectsLoader')
  },
  
  // Notifications
  notifications: {
    container: document.getElementById('notificationContainer'),
    text: document.getElementById('notificationText')
  }
};

/**
 * Display a notification to the user
 * @param {string} message - Text to display
 * @param {string} type - Type of notification (success, error, info, warning)
 * @param {number} duration - Time in ms to show the notification (default: 3000)
 */
function showNotification(message, type = 'info', duration = 3000) {
  const container = UI.notifications.container;
  const text = UI.notifications.text;
  
  // Set the message
  text.textContent = message;
  
  // Clear previous classes
  container.className = 'notification';
  
  // Add appropriate class
  container.classList.add(`notification-${type}`);
  
  // Show notification
  container.classList.add('show');
  
  // Hide after duration
  setTimeout(() => {
    container.classList.remove('show');
  }, duration);
}

/**
 * Show API success message
 * @param {string} message - Success message to show
 */
function showApiSuccess(message) {
  showNotification(message, 'success');
}

/**
 * Show API error message
 * @param {string} message - Error message to show
 */
function showApiError(message) {
  showNotification(message, 'error');
}

/**
 * Toggle loader visibility
 * @param {string} loaderId - ID of the loader element 
 * @param {boolean} show - Whether to show or hide the loader
 */
function toggleLoader(loaderId, show) {
  const loader = document.getElementById(loaderId);
  if (loader) {
    if (show) {
      loader.classList.remove('hidden');
    } else {
      loader.classList.add('hidden');
    }
  }
}

/**
 * Navigate between sections
 * @param {string} sectionId - ID of the section to show
 */
function navigateTo(sectionId) {
  // Hide all sections
  Object.values(UI.sections).forEach(section => {
    if (section) {
      section.classList.add('hidden');
    }
  });
  
  // Show requested section
  const targetSection = document.getElementById(sectionId);
  if (targetSection) {
    targetSection.classList.remove('hidden');
    // Scroll to top
    window.scrollTo(0, 0);
  }
}

/**
 * Render the list of subjects
 * @param {Array} subjects - Array of subject objects
 * @param {string} containerId - ID of the container element
 */
function renderSubjectsList(subjects, containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  // Clear container
  container.innerHTML = '';
  
  if (subjects.length === 0) {
    container.innerHTML = '<p class="empty-state">No subjects added yet. Add subjects by uploading a syllabus or manually.</p>';
    return;
  }
  
  // Create list
  const list = document.createElement('ul');
  list.className = 'subject-list';
  
  // Add each subject
  subjects.forEach((subject, index) => {
    const item = document.createElement('li');
    item.className = 'subject-item';
    
    // Build HTML
    let html = `
      <div class="subject-header">
        <h3>${subject.name}</h3>
        <div class="subject-actions">
          <button class="btn-icon edit-subject" data-index="${index}">
            <i class="fas fa-edit"></i>
          </button>
          <button class="btn-icon delete-subject" data-index="${index}">
            <i class="fas fa-trash"></i>
          </button>
        </div>
      </div>
    `;
    
    // Add topics if available
    if (subject.topics && subject.topics.length > 0) {
      html += '<h4>Topics:</h4><ul class="topic-list">';
      subject.topics.forEach(topic => {
        html += `<li>${topic}</li>`;
      });
      html += '</ul>';
    }
    
    // Add importance/weight if available
    if (subject.importance || subject.weight) {
      const value = subject.importance || subject.weight || 3;
      html += `
        <div class="subject-importance">
          <span>Importance:</span>
          <div class="importance-level">
            ${generateImportanceStars(value)}
          </div>
        </div>
      `;
    }
    
    item.innerHTML = html;
    list.appendChild(item);
  });
  
  container.appendChild(list);
  
  // Attach event listeners
  attachSubjectListEvents(containerId);
}

/**
 * Generate stars for importance level
 * @param {number} level - Importance level (1-5)
 * @returns {string} HTML string with star icons
 */
function generateImportanceStars(level) {
  level = Math.min(5, Math.max(1, level)); // Ensure between 1-5
  let stars = '';
  
  for (let i = 1; i <= 5; i++) {
    if (i <= level) {
      stars += '<i class="fas fa-star"></i>';
    } else {
      stars += '<i class="far fa-star"></i>';
    }
  }
  
  return stars;
}

/**
 * Attach event listeners to subject list items
 * @param {string} containerId - ID of the container element
 */
function attachSubjectListEvents(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  
  // Edit subject event
  const editButtons = container.querySelectorAll('.edit-subject');
  editButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      const index = parseInt(e.currentTarget.dataset.index);
      openEditSubjectModal(index, containerId === 'extractedSubjectsList' ? 'extracted' : 'manual');
    });
  });
  
  // Delete subject event
  const deleteButtons = container.querySelectorAll('.delete-subject');
  deleteButtons.forEach(button => {
    button.addEventListener('click', (e) => {
      const index = parseInt(e.currentTarget.dataset.index);
      confirmDeleteSubject(index, containerId === 'extractedSubjectsList' ? 'extracted' : 'manual');
    });
  });
}

/**
 * Open the modal to edit a subject
 * @param {number} index - Index of the subject to edit
 * @param {string} type - Type of subject ('extracted' or 'manual')
 */
function openEditSubjectModal(index, type) {
  // Get the subject from the appropriate array
  const subject = type === 'extracted' ? extractedSubjects[index] : manualSubjects[index];
  if (!subject) return;
  
  // Populate modal with subject data
  const modal = document.getElementById('editSubjectModal');
  const nameInput = document.getElementById('editSubjectName');
  const importanceInput = document.getElementById('editSubjectImportance');
  const topicsContainer = document.getElementById('editSubjectTopics');
  
  nameInput.value = subject.name;
  importanceInput.value = subject.importance || subject.weight || 3;
  
  // Clear topics container
  topicsContainer.innerHTML = '';
  
  // Add existing topics
  if (subject.topics && subject.topics.length > 0) {
    subject.topics.forEach(topic => {
      addTopicInput(topic);
    });
  } else {
    // Add one empty topic input
    addTopicInput('');
  }
  
  // Set data attributes for form handling
  const form = document.getElementById('editSubjectForm');
  form.dataset.index = index;
  form.dataset.type = type;
  
  // Show modal
  modal.classList.remove('hidden');
}

/**
 * Add a topic input field to the edit form
 * @param {string} value - Initial value for the topic
 */
function addTopicInput(value = '') {
  const container = document.getElementById('editSubjectTopics');
  const topicId = 'topic-' + Date.now() + Math.floor(Math.random() * 1000);
  
  const topicDiv = document.createElement('div');
  topicDiv.className = 'topic-input-group';
  topicDiv.innerHTML = `
    <input type="text" id="${topicId}" value="${value}" placeholder="Topic name" class="topic-input">
    <button type="button" class="btn-icon remove-topic">
      <i class="fas fa-times"></i>
    </button>
  `;
  
  container.appendChild(topicDiv);
  
  // Add event listener to remove button
  const removeButton = topicDiv.querySelector('.remove-topic');
  removeButton.addEventListener('click', () => {
    topicDiv.remove();
  });
}

/**
 * Save the edited subject
 */
function saveEditedSubject() {
  const form = document.getElementById('editSubjectForm');
  const index = parseInt(form.dataset.index);
  const type = form.dataset.type;
  
  // Get values
  const name = document.getElementById('editSubjectName').value;
  const importance = parseInt(document.getElementById('editSubjectImportance').value);
  
  // Get topics
  const topics = [];
  const topicInputs = document.querySelectorAll('.topic-input');
  topicInputs.forEach(input => {
    if (input.value.trim()) {
      topics.push(input.value.trim());
    }
  });
  
  // Validate
  if (!name) {
    showApiError('Subject name is required');
    return;
  }
  
  // Create updated subject object
  const updatedSubject = {
    name,
    topics,
    importance
  };
  
  // Update appropriate array
  if (type === 'extracted') {
    extractedSubjects[index] = updatedSubject;
    // Save to storage
    saveExtractedSubjects(extractedSubjects);
    // Re-render
    renderSubjectsList(extractedSubjects, 'extractedSubjectsList');
  } else {
    manualSubjects[index] = updatedSubject;
    // Save to storage
    saveManualSubjects(manualSubjects);
    // Re-render
    renderSubjectsList(manualSubjects, 'manualSubjectsList');
  }
  
  // Close modal
  closeModal('editSubjectModal');
  
  // Show success message
  showApiSuccess('Subject updated successfully');
}

/**
 * Confirm and delete a subject
 * @param {number} index - Index of the subject to delete
 * @param {string} type - Type of subject ('extracted' or 'manual')
 */
function confirmDeleteSubject(index, type) {
  if (confirm('Are you sure you want to delete this subject?')) {
    if (type === 'extracted') {
      extractedSubjects.splice(index, 1);
      // Save to storage
      saveExtractedSubjects(extractedSubjects);
      // Re-render
      renderSubjectsList(extractedSubjects, 'extractedSubjectsList');
    } else {
      manualSubjects.splice(index, 1);
      // Save to storage
      saveManualSubjects(manualSubjects);
      // Re-render
      renderSubjectsList(manualSubjects, 'manualSubjectsList');
    }
    
    showApiSuccess('Subject deleted successfully');
  }
}

/**
 * Close a modal
 * @param {string} modalId - ID of the modal to close
 */
function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) {
    modal.classList.add('hidden');
  }
}

/**
 * Render the study plan
 * @param {Object} plan - The study plan object
 */
function renderStudyPlan(plan) {
  const container = document.getElementById('studyPlanContent');
  if (!container) return;
  
  // Clear container
  container.innerHTML = '';
  
  // Plan metadata
  const metaHtml = `
    <div class="plan-meta">
      <div class="meta-item">
        <i class="fas fa-calendar-alt"></i>
        <span>Start: ${plan.startDate || plan.meta?.start_date}</span>
      </div>
      <div class="meta-item">
        <i class="fas fa-calendar-check"></i>
        <span>End: ${plan.endDate || plan.meta?.end_date}</span>
      </div>
      <div class="meta-item">
        <i class="fas fa-graduation-cap"></i>
        <span>Total Days: ${plan.totalDays || plan.meta?.total_days}</span>
      </div>
      <div class="meta-item">
        <i class="fas fa-book"></i>
        <span>Subjects: ${plan.subjects?.length || plan.meta?.subjects?.length}</span>
      </div>
    </div>
  `;
  
  container.innerHTML = metaHtml;
  
  // Progress bar
  const progressPercentage = calculatePlanProgress(plan);
  const progressHtml = `
    <div class="progress-container">
      <div class="progress-label">Overall Progress</div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: ${progressPercentage}%"></div>
      </div>
      <div class="progress-percentage">${progressPercentage}%</div>
    </div>
  `;
  
  container.innerHTML += progressHtml;
  
  // Daily plans
  const dailyPlansHtml = `<h3>Daily Study Schedule</h3><div class="daily-plans">`;
  container.innerHTML += dailyPlansHtml;
  
  const dailyPlansContainer = document.createElement('div');
  dailyPlansContainer.className = 'daily-plans-container';
  
  // Get the daily plans in the correct format
  const dailyPlans = plan.dailyPlan || Object.entries(plan.daily_plans || {}).map(([date, dayPlan]) => {
    return {
      date,
      ...dayPlan
    };
  }).sort((a, b) => new Date(a.date) - new Date(b.date));
  
  // Add each daily plan
  dailyPlans.forEach(day => {
    const dayElement = document.createElement('div');
    dayElement.className = 'day-plan';
    
    if (day.completed) {
      dayElement.classList.add('completed');
    }
    
    // Get date in readable format
    const dateObj = new Date(day.date);
    const formattedDate = dateObj.toLocaleDateString('en-US', { 
      weekday: 'short', 
      month: 'short', 
      day: 'numeric' 
    });
    
    // Day header
    let dayHeader = `
      <div class="day-header">
        <div class="day-number">Day ${day.day || day.day_number}</div>
        <div class="day-date">${formattedDate}</div>
    `;
    
    // Add completion toggle
    dayHeader += `
        <label class="day-complete-toggle">
          <input type="checkbox" class="day-complete-checkbox" 
            data-date="${day.date}" ${day.completed ? 'checked' : ''}>
          <span class="checkmark"></span>
          <span class="toggle-label">Complete</span>
        </label>
      </div>
    `;
    
    // Begin day content
    let dayContent = `<div class="day-content">`;
    
    // Add subjects
    const daySubjects = day.subjects || [];
    daySubjects.forEach(subjectItem => {
      const subjectName = subjectItem.subject || subjectItem.name;
      const topics = subjectItem.topics || [];
      const duration = subjectItem.duration_minutes 
        ? `${Math.round(subjectItem.duration_minutes / 60 * 10) / 10} hours` 
        : `${subjectItem.duration || 2} hours`;
      
      dayContent += `
        <div class="day-subject">
          <h4>${subjectName}</h4>
          <div class="subject-duration"><i class="far fa-clock"></i> ${duration}</div>
      `;
      
      // Add topics if available
      if (topics.length > 0) {
        dayContent += `<div class="subject-topics"><h5>Topics:</h5><ul>`;
        topics.forEach(topic => {
          dayContent += `<li>${topic}</li>`;
        });
        dayContent += `</ul></div>`;
      }
      
      // Add resources if available
      const resources = subjectItem.resources || [];
      if (resources.length > 0) {
        dayContent += `<div class="subject-resources"><h5>Resources:</h5><ul>`;
        resources.forEach(resource => {
          dayContent += `
            <li>
              <a href="${resource.url}" target="_blank" class="resource-link">
                <i class="fas fa-${getResourceIcon(resource.type)}"></i>
                ${resource.title}
              </a>
            </li>
          `;
        });
        dayContent += `</ul></div>`;
      }
      
      dayContent += `</div>`;
    });
    
    // Add notes if available
    if (day.notes) {
      dayContent += `
        <div class="day-notes">
          <h5><i class="fas fa-sticky-note"></i> Notes:</h5>
          <p>${day.notes}</p>
        </div>
      `;
    }
    
    dayContent += `</div>`; // Close day-content
    
    dayElement.innerHTML = dayHeader + dayContent;
    dailyPlansContainer.appendChild(dayElement);
  });
  
  container.appendChild(dailyPlansContainer);
  
  // Attach event listeners for completion checkboxes
  attachDayCompletionEvents();
}

/**
 * Get appropriate icon for resource type
 * @param {string} type - Resource type
 * @returns {string} - Font Awesome icon name
 */
function getResourceIcon(type) {
  const icons = {
    video: 'video',
    article: 'newspaper',
    book: 'book',
    exercise: 'dumbbell',
    quiz: 'question-circle',
    notes: 'sticky-note',
    interactive: 'laptop-code',
    course: 'graduation-cap',
    search: 'search'
  };
  
  return icons[type] || 'link';
}

/**
 * Calculate overall plan progress percentage
 * @param {Object} plan - The study plan object
 * @returns {number} - Progress percentage
 */
function calculatePlanProgress(plan) {
  let totalDays = 0;
  let completedDays = 0;
  
  // Check if using new or old format
  if (plan.dailyPlan) {
    totalDays = plan.dailyPlan.length;
    completedDays = plan.dailyPlan.filter(day => day.completed).length;
  } else if (plan.daily_plans) {
    totalDays = Object.keys(plan.daily_plans).length;
    completedDays = Object.values(plan.daily_plans).filter(day => day.completed).length;
  } else if (plan.meta) {
    totalDays = plan.meta.total_days;
    completedDays = plan.meta.completed_days || 0;
  }
  
  if (totalDays === 0) return 0;
  return Math.round((completedDays / totalDays) * 100);
}

/**
 * Attach event listeners to day completion checkboxes
 */
function attachDayCompletionEvents() {
  const checkboxes = document.querySelectorAll('.day-complete-checkbox');
  checkboxes.forEach(checkbox => {
    checkbox.addEventListener('change', (e) => {
      const date = e.target.dataset.date;
      const completed = e.target.checked;
      
      // Update study plan
      updateDayCompletion(date, completed);
      
      // Toggle completed class on parent element
      const dayElement = e.target.closest('.day-plan');
      if (dayElement) {
        if (completed) {
          dayElement.classList.add('completed');
        } else {
          dayElement.classList.remove('completed');
        }
      }
      
      // Update progress display
      const progressPercentage = calculatePlanProgress(studyPlan);
      const progressFill = document.querySelector('.progress-fill');
      const progressText = document.querySelector('.progress-percentage');
      
      if (progressFill) {
        progressFill.style.width = `${progressPercentage}%`;
      }
      
      if (progressText) {
        progressText.textContent = `${progressPercentage}%`;
      }
      
      // Save progress to storage
      saveProgress(studyPlan);
    });
  });
}

/**
 * Update a day's completion status in the study plan
 * @param {string} date - Date string of the day to update
 * @param {boolean} completed - New completion status
 */
function updateDayCompletion(date, completed) {
  // Check if using new or old format
  if (studyPlan.dailyPlan) {
    const day = studyPlan.dailyPlan.find(d => d.date === date);
    if (day) {
      day.completed = completed;
    }
    
    // Update meta counters
    studyPlan.completedDays = studyPlan.dailyPlan.filter(d => d.completed).length;
  } else if (studyPlan.daily_plans && studyPlan.daily_plans[date]) {
    studyPlan.daily_plans[date].completed = completed;
    
    // Update meta counters
    studyPlan.meta.completed_days = Object.values(studyPlan.daily_plans)
      .filter(d => d.completed).length;
  }
}

/**
 * Render user progress statistics
 */
function renderProgressStats() {
  const container = document.getElementById('progressStatsContainer');
  if (!container || !studyPlan) return;
  
  // Calculate statistics
  const totalDays = studyPlan.totalDays || studyPlan.meta?.total_days || 0;
  const completedDays = calculateCompletedDays(studyPlan);
  const progressPercentage = totalDays > 0 ? Math.round((completedDays / totalDays) * 100) : 0;
  
  // Current day of plan
  const startDate = new Date(studyPlan.startDate || studyPlan.meta?.start_date);
  const today = new Date();
  const currentDay = Math.ceil((today - startDate) / (1000 * 60 * 60 * 24)) + 1;
  const onTrack = completedDays >= Math.min(currentDay, totalDays);
  
  // Build HTML
  const html = `
    <div class="stat-card">
      <div class="stat-icon"><i class="fas fa-calendar-check"></i></div>
      <div class="stat-content">
        <div class="stat-value">${completedDays}/${totalDays}</div>
        <div class="stat-label">Days Completed</div>
      </div>
    </div>
    
    <div class="stat-card">
      <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
      <div class="stat-content">
        <div class="stat-value">${progressPercentage}%</div>
        <div class="stat-label">Overall Progress</div>
      </div>
    </div>
    
    <div class="stat-card">
      <div class="stat-icon">
        <i class="fas fa-${onTrack ? 'thumbs-up' : 'exclamation-circle'}"></i>
      </div>
      <div class="stat-content">
        <div class="stat-value">${onTrack ? 'On Track' : 'Behind Schedule'}</div>
        <div class="stat-label">Current Status</div>
      </div>
    </div>
    
    <div class="stat-card">
      <div class="stat-icon"><i class="fas fa-calendar-day"></i></div>
      <div class="stat-content">
        <div class="stat-value">Day ${Math.min(currentDay, totalDays)}</div>
        <div class="stat-label">Current Day</div>
      </div>
    </div>
  `;
  
  container.innerHTML = html;
}

/**
 * Calculate total completed days in a study plan
 * @param {Object} plan - The study plan object
 * @returns {number} - Number of completed days
 */
function calculateCompletedDays(plan) {
  if (!plan) return 0;
  
  // Check which format is being used
  if (plan.dailyPlan) {
    return plan.dailyPlan.filter(day => day.completed).length;
  } else if (plan.daily_plans) {
    return Object.values(plan.daily_plans).filter(day => day.completed).length;
  } else if (plan.meta && plan.meta.completed_days !== undefined) {
    return plan.meta.completed_days;
  }
  
  return 0;
}

// Export UI functions to global scope for HTML access
window.showNotification = showNotification;
window.showApiSuccess = showApiSuccess;
window.showApiError = showApiError;
window.toggleLoader = toggleLoader;
window.navigateTo = navigateTo;
window.renderSubjectsList = renderSubjectsList;
window.openEditSubjectModal = openEditSubjectModal;
window.addTopicInput = addTopicInput;
window.saveEditedSubject = saveEditedSubject;
window.confirmDeleteSubject = confirmDeleteSubject;
window.closeModal = closeModal;
window.renderStudyPlan = renderStudyPlan;
window.renderProgressStats = renderProgressStats;