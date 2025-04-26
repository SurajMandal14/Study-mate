/**
 * API Module
 * Handles external API calls and service integrations
 */

// Base URL for Grok API
const GROK_API_BASE_URL = 'https://teachnook.com/techsnap/chat/';

// Verify the API key provided by the user
async function verifyApiKey() {
  const apiKeyInput = document.getElementById('grokApiKey');
  const apiKey = apiKeyInput.value.trim();
  
  if (!apiKey) {
    showApiError('Please enter a valid API key');
    return;
  }
  
  try {
    // Show loading state
    document.getElementById('grokSaveBtn').innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> Verifying';
    document.getElementById('grokSaveBtn').disabled = true;
    
    // Simulate API verification (replace with actual API call when endpoint is available)
    const result = await simulateApiVerification(apiKey);
    
    if (result.success) {
      // Store API key (in memory only - would use more secure storage in production)
      grokApiKey = apiKey;
      grokApiAuthenticated = true;
      
      // Update UI
      showApiSuccess('API key verified successfully');
      updateApiAuthUI(true);
      
      // Hide the API input section
      setTimeout(() => {
        document.getElementById('apiKeySection').classList.add('hidden');
        document.getElementById('uploadSection').classList.remove('hidden');
      }, 1000);
    } else {
      showApiError(result.message || 'API verification failed');
      grokApiAuthenticated = false;
      updateApiAuthUI(false);
    }
  } catch (error) {
    console.error('API verification error:', error);
    showApiError('Connection error. Please try again.');
    grokApiAuthenticated = false;
    updateApiAuthUI(false);
  } finally {
    // Reset button state
    document.getElementById('grokSaveBtn').innerHTML = 'Verify API Key';
    document.getElementById('grokSaveBtn').disabled = false;
  }
}

// Simulate API verification (replace with actual API call)
async function simulateApiVerification(apiKey) {
  return new Promise((resolve) => {
    setTimeout(() => {
      // For demo purposes, we're accepting any key that's at least 8 chars
      // In production, this would validate with the actual Grok API
      const isValid = apiKey.length >= 8;
      resolve({
        success: isValid,
        message: isValid ? 'API key verified' : 'Invalid API key'
      });
    }, 1500);
  });
}

// Handle PDF upload and processing
async function handlePdfUpload(event) {
  const file = event.target.files[0];
  if (!file) return;
  
  // Validate file is PDF
  if (file.type !== 'application/pdf') {
    showApiError('Please upload a PDF file');
    return;
  }
  
  try {
    // Show processing overlay
    showPdfProcessing(true);
    
    // Read the file as base64
    const base64Data = await readFileAsBase64(file);
    
    // Make API call to process PDF (simulated)
    const result = await processPdfWithGrok(base64Data, file.name);
    
    if (result.success) {
      // Store extracted subjects
      extractedSubjects = result.subjects;
      
      // Display subjects from PDF
      renderExtractedSubjects(extractedSubjects);
      
      // Show the subjects panel
      document.getElementById('extractedSubjectsSection').classList.remove('hidden');
      
      // Scroll to the extracted subjects
      document.getElementById('extractedSubjectsSection').scrollIntoView({
        behavior: 'smooth'
      });
      
      // Show success message
      showApiSuccess('PDF analyzed successfully');
    } else {
      showApiError(result.message || 'Failed to analyze PDF');
    }
  } catch (error) {
    console.error('PDF processing error:', error);
    showApiError('Error processing PDF. Please try again.');
  } finally {
    showPdfProcessing(false);
  }
}

// Read a file as base64
function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // Get base64 string (remove data URL prefix)
      const base64 = reader.result.split(',')[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// Process PDF with Grok API (simulated)
async function processPdfWithGrok(base64Data, filename) {
  // In a real implementation, this would call the actual Grok API
  return new Promise((resolve) => {
    setTimeout(() => {
      // Simulate processing
      const simulatedSubjects = [
        {
          name: 'Data Structures',
          topics: ['Arrays', 'Linked Lists', 'Trees', 'Graphs', 'Hash Tables'],
          weight: 5,
          confidence: 0.92
        },
        {
          name: 'Algorithms',
          topics: ['Sorting', 'Searching', 'Dynamic Programming', 'Greedy Algorithms'],
          weight: 4,
          confidence: 0.89
        },
        {
          name: 'Database Systems',
          topics: ['SQL', 'Normalization', 'Transactions', 'Indexing'],
          weight: 3,
          confidence: 0.85
        },
        {
          name: 'Operating Systems',
          topics: ['Process Management', 'Memory Management', 'File Systems'],
          weight: 2,
          confidence: 0.81
        }
      ];
      
      resolve({
        success: true,
        subjects: simulatedSubjects,
        message: 'PDF analyzed successfully'
      });
    }, 3000);
  });
}

// Generate study plan using an external API
async function generateStudyPlan(subjects, examDate, preferences = {}) {
  try {
    // For now, simulate API call with local implementation
    return simulateStudyPlanGeneration(subjects, examDate, preferences);
    
    // Uncomment below when real API is available
    /*
    const response = await fetch(`${API_CONFIG.BASE_URL}/study-plan/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        subjects,
        examDate,
        preferences
      })
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
    */
  } catch (error) {
    console.error('Error generating study plan:', error);
    throw error;
  }
}

// Extract subjects from syllabus PDF
async function extractSubjectsFromPDF(fileData) {
  try {
    // For now, simulate API call with local implementation
    return simulateSubjectExtraction(fileData);
    
    // Uncomment below when real API is available
    /*
    const formData = new FormData();
    formData.append('file', fileData);
    
    const response = await fetch(`${API_CONFIG.BASE_URL}/extract/subjects`, {
      method: 'POST',
      body: formData
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
    */
  } catch (error) {
    console.error('Error extracting subjects:', error);
    throw error;
  }
}

// Get learning resources for a subject
async function getSubjectResources(subjectName) {
  try {
    // For now, simulate API call with local implementation
    return simulateResourceRetrieval(subjectName);
    
    // Uncomment below when real API is available
    /*
    const response = await fetch(`${API_CONFIG.BASE_URL}/resources?subject=${encodeURIComponent(subjectName)}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json'
      }
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
    */
  } catch (error) {
    console.error('Error fetching resources:', error);
    throw error;
  }
}

// Send user feedback to the server
async function sendFeedback(feedbackData) {
  try {
    // For now, log to console
    console.log('Feedback received:', feedbackData);
    return { success: true, message: 'Feedback recorded' };
    
    // Uncomment below when real API is available
    /*
    const response = await fetch(`${API_CONFIG.BASE_URL}/feedback`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(feedbackData)
    });
    
    if (!response.ok) {
      throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
    */
  } catch (error) {
    console.error('Error sending feedback:', error);
    throw error;
  }
}

// Simulate study plan generation (local implementation)
function simulateStudyPlanGeneration(subjects, examDate, preferences) {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Parse the exam date
      const targetDate = new Date(examDate);
      const currentDate = new Date();
      
      // Calculate days until exam
      const daysUntilExam = Math.max(1, Math.ceil((targetDate - currentDate) / (1000 * 60 * 60 * 24)));
      
      // Determine study plan parameters
      const minDaysPerSubject = Math.max(1, Math.floor(daysUntilExam / subjects.length));
      const extraDays = daysUntilExam - (minDaysPerSubject * subjects.length);
      
      // Sort subjects by importance or difficulty
      const sortedSubjects = [...subjects].sort((a, b) => 
        (b.importance || 5) - (a.importance || 5)
      );
      
      // Assign extra days to more important subjects
      const subjectsWithDays = sortedSubjects.map((subject, index) => {
        const extraDaysForThisSubject = index < extraDays ? 1 : 0;
        return {
          ...subject,
          daysAllocated: minDaysPerSubject + extraDaysForThisSubject
        };
      });
      
      // Generate the actual plan
      const studyPlan = {
        startDate: currentDate.toISOString().split('T')[0],
        endDate: examDate,
        totalDays: daysUntilExam,
        subjects: subjectsWithDays,
        dailyPlan: []
      };
      
      // Create daily plan by distributing subjects
      let currentDateIter = new Date(currentDate);
      let subjectIndex = 0;
      
      for (let day = 0; day < daysUntilExam; day++) {
        const dateString = currentDateIter.toISOString().split('T')[0];
        const subject = subjectsWithDays[subjectIndex % subjectsWithDays.length];
        
        studyPlan.dailyPlan.push({
          date: dateString,
          day: day + 1,
          subjects: [{
            name: subject.name,
            topics: subject.topics ? subject.topics.slice(0, 3) : [],
            duration: preferences.studyHoursPerDay || 2
          }]
        });
        
        // Move to next subject after covering all days for current subject
        if ((day + 1) % subject.daysAllocated === 0) {
          subjectIndex++;
        }
        
        // Increment date
        currentDateIter.setDate(currentDateIter.getDate() + 1);
      }
      
      resolve(studyPlan);
    }, 1500); // Simulate network delay
  });
}

// Simulate subject extraction from syllabus (local implementation)
function simulateSubjectExtraction(fileData) {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Default subjects if actual extraction is not possible
      const extractedSubjects = [
        {
          name: "Mathematics",
          topics: ["Algebra", "Calculus", "Statistics", "Geometry"],
          importance: 5
        },
        {
          name: "Physics",
          topics: ["Mechanics", "Electromagnetism", "Thermodynamics", "Optics"],
          importance: 4
        },
        {
          name: "Chemistry",
          topics: ["Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry", "Analytical Methods"],
          importance: 4
        },
        {
          name: "Biology",
          topics: ["Cell Biology", "Genetics", "Ecology", "Physiology"],
          importance: 3
        },
        {
          name: "Computer Science",
          topics: ["Algorithms", "Data Structures", "Operating Systems", "Networks"],
          importance: 5
        }
      ];
      
      resolve({
        success: true,
        subjects: extractedSubjects,
        message: "Subjects extracted successfully"
      });
    }, 2000); // Simulate network delay
  });
}

// Simulate resource retrieval (local implementation)
function simulateResourceRetrieval(subjectName) {
  return new Promise((resolve) => {
    setTimeout(() => {
      // Default resources based on subject
      const subjectResources = {
        "Mathematics": [
          { title: "Khan Academy - Math", url: "https://www.khanacademy.org/math", type: "video" },
          { title: "Paul's Online Math Notes", url: "https://tutorial.math.lamar.edu/", type: "notes" },
          { title: "MIT OpenCourseWare - Mathematics", url: "https://ocw.mit.edu/courses/mathematics/", type: "course" }
        ],
        "Physics": [
          { title: "Physics Classroom", url: "https://www.physicsclassroom.com/", type: "interactive" },
          { title: "Feynman Lectures on Physics", url: "https://www.feynmanlectures.caltech.edu/", type: "notes" },
          { title: "Khan Academy - Physics", url: "https://www.khanacademy.org/science/physics", type: "video" }
        ],
        "Chemistry": [
          { title: "Chemistry LibreTexts", url: "https://chem.libretexts.org/", type: "notes" },
          { title: "Khan Academy - Chemistry", url: "https://www.khanacademy.org/science/chemistry", type: "video" },
          { title: "Royal Society of Chemistry", url: "https://edu.rsc.org/", type: "interactive" }
        ],
        "Biology": [
          { title: "Khan Academy - Biology", url: "https://www.khanacademy.org/science/biology", type: "video" },
          { title: "Biology Online", url: "https://www.biology-online.org/", type: "notes" },
          { title: "iBiology", url: "https://www.ibiology.org/", type: "video" }
        ],
        "Computer Science": [
          { title: "Codecademy", url: "https://www.codecademy.com/", type: "interactive" },
          { title: "GeeksforGeeks", url: "https://www.geeksforgeeks.org/", type: "notes" },
          { title: "freeCodeCamp", url: "https://www.freecodecamp.org/", type: "interactive" }
        ]
      };
      
      // Return resources for the specified subject or default ones
      const resources = subjectResources[subjectName] || [
        { title: "Google Scholar", url: "https://scholar.google.com/", type: "search" },
        { title: "MIT OpenCourseWare", url: "https://ocw.mit.edu/", type: "course" },
        { title: "Khan Academy", url: "https://www.khanacademy.org/", type: "video" }
      ];
      
      resolve({
        success: true,
        resources: resources,
        subject: subjectName
      });
    }, 1000); // Simulate network delay
  });
}

// Show API error message
function showApiError(message) {
  const errorEl = document.getElementById('apiErrorMsg');
  errorEl.textContent = message;
  errorEl.classList.remove('hidden');
  errorEl.classList.remove('text-green-600');
  errorEl.classList.add('text-red-600');
  
  // Hide after 5 seconds
  setTimeout(() => {
    errorEl.classList.add('hidden');
  }, 5000);
}

// Show API success message
function showApiSuccess(message) {
  const errorEl = document.getElementById('apiErrorMsg');
  errorEl.textContent = message;
  errorEl.classList.remove('hidden');
  errorEl.classList.remove('text-red-600');
  errorEl.classList.add('text-green-600');
  
  // Hide after 5 seconds
  setTimeout(() => {
    errorEl.classList.add('hidden');
  }, 5000);
}

// Export functions to global scope for HTML access
window.verifyApiKey = verifyApiKey;
window.handlePdfUpload = handlePdfUpload;
window.generateStudyPlan = generateStudyPlan;
window.extractSubjectsFromPDF = extractSubjectsFromPDF;
window.getSubjectResources = getSubjectResources;
window.sendFeedback = sendFeedback;
window.showApiError = showApiError;
window.showApiSuccess = showApiSuccess; 