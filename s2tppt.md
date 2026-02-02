================================================================================
POWERPOINT PRESENTATION - FINAL SLIDE CONTENT
Ready for Gemini PPT Generation
================================================================================

Date: February 2, 2026
Total Slides: 15
Format: Concise, slide-ready content

================================================================================
SLIDE 1: TITLE SLIDE
================================================================================

Title: Technical Innovation & Progress Update

Subtitle:
• Speech-to-Text with Speaker Diarization
• Fuzzy Matching Tool (Application-as-Recipe)
• Topic Mining Tool (Application-as-Recipe)

Footer:
Presented by: [Your Name]
Date: February 2, 2026

---

DESIGN NOTES:

- Use clean corporate template
- Add icons: 🎤 Microphone, 🔗 Network, 📊 Analytics
- Company logo in corner

---

================================================================================
SLIDE 2: AGENDA
================================================================================

Title: Today's Agenda

Content:

**PROJECT 1: Speech-to-Text System**
✓ Offline audio transcription
✓ Rule-based speaker diarization
✓ Innovation: Zero API costs

**PROJECT 2: Fuzzy Matching Tool**
✓ Webapp → Application-as-Recipe conversion
✓ Integration into Dataiku
✓ Collaboration with Shubham

**PROJECT 3: Topic Mining Tool**
✓ Streamlit App → Recipe conversion
✓ Code segregation process
✓ AAR framework implementation

**Additional Updates**
✓ Smart Scheduler Documentation (80%)
✓ Innovation Reports Submitted

---

DESIGN NOTES:

- Use 3 colored boxes for projects
- Icons for each project
- Timeline/progress bar at bottom

---

================================================================================
SLIDE 3: SPEECH-TO-TEXT OVERVIEW
================================================================================

Title: Speech-to-Text with Speaker Diarization

**The Problem:**
❌ Google/Azure APIs → $0.006-0.024/min
❌ Privacy concerns → Cloud uploads
❌ Vosk offline → 20-30% error rate

**Our Solution:**
✅ 100% offline processing
✅ 5-12% WER (Word Error Rate)
✅ Speaker identification included
✅ Zero operational costs

**Technology Stack:**
• Whisper (HuggingFace) - Transcription
• MFCC + Clustering - Diarization
• Flask REST API - Backend
• librosa/scipy - Audio processing

**Key Metrics:**
→ Accuracy: 5-12% WER
→ Speakers: Auto-detects 2-6
→ Speed: 1-2x real-time
→ Cost: $0 vs $4-17K/year

---

DESIGN NOTES:

- Before/After comparison table
- Technology stack icons
- Metrics in colored boxes

---

================================================================================
SLIDE 4: SPEECH-TO-TEXT ARCHITECTURE
================================================================================

Title: How It Works - Technical Pipeline

**FLOWCHART PROMPT:**
"Create a vertical flowchart with 4 stages:

Stage 1: Audio Preprocessing

- Noise Reduction (STFT)
- Resampling to 16kHz
- High-Pass Filtering
  [Use blue color]

Stage 2: Whisper Transcription

- 25s chunks, 5s overlap
- GPU/CPU processing
- SafeTensors format
  [Use green color]

Stage 3: Speaker Diarization

- Silence Detection (RMS)
- MFCC Feature Extraction (20 coefficients)
- Agglomerative Clustering
  [Use orange color]

Stage 4: Output

- Speaker Labels
- Punctuation-based splits
- Timestamped transcript
  [Use purple color]

Add arrows between stages. Include icons for each component. Make it professional and clean."

**Innovation Highlights:**
🔹 Hybrid approach: Audio + Linguistic signals
🔹 Auto-detection: No manual configuration
🔹 Offline: Complete privacy

---

DESIGN NOTES:

- Use generated flowchart as main visual
- Highlight boxes on right side for innovations

---

================================================================================
SLIDE 5: RULE-BASED DIARIZATION
================================================================================

Title: Intelligent Rule-Based Speaker Identification

**Why Rule-Based?**
• Avoid Pyannote.audio API dependency
• 100% offline processing
• Explainable logic (not black box)

**Core Rule: Punctuation Splitting**
? → Likely speaker change
. → Potential speaker change
! → Potential speaker change

**The Problem:**
❌ "The bank? It's not far."
→ False split (same person!)

**The Solution: 4 Continuation Guards**

**Guard #1: Silence Gap**
Gap < 0.6s → Same speaker

**Guard #2: Continuation Words**
because, so, and, but, also → Same speaker

**Guard #3: First-Person Pronouns**
I, we, my, our → Same speaker

**Guard #4: Sentence Length**
Both > 10 words → Same speaker (monologue)

**Results:**
📊 Punctuation only: 60-65%
📊 With guards: 85-90%
📊 Target (word-level): 90%+

---

DESIGN NOTES:

- 4 guard boxes with icons
- Before/After accuracy comparison chart
- Visual example of guard system

---

================================================================================
SLIDE 6: RULE SYSTEM FLOWCHART
================================================================================

Title: Continuation Guard Decision Logic

**FLOWCHART PROMPT:**
"Create a decision tree flowchart:

START: Sentence ends with [? . !]
↓
[Diamond] Gap < 0.6 seconds?
→ YES: Keep Same Speaker [Green box, EXIT]
→ NO: Continue ↓

[Diamond] Starts with continuation word?
→ YES: Keep Same Speaker [Green box, EXIT]
→ NO: Continue ↓

[Diamond] Contains I/we/my/our?
→ YES: Keep Same Speaker [Green box, EXIT]
→ NO: Continue ↓

[Diamond] Both sentences > 10 words?
→ YES: Keep Same Speaker [Green box, EXIT]
→ NO: Switch Speaker [Red box]

Use diamond shapes for decisions, green rectangles for 'Keep Same', red rectangle for 'Switch'. Add example sentence at each decision point."

---

DESIGN NOTES:

- Use generated flowchart
- Add title "How Guards Prevent False Splits"

---

================================================================================
SLIDE 7: SPEECH-TO-TEXT STATUS
================================================================================

Title: Current Status & Next Steps

**What's Working ✅**
✓ Transcription: 5-12% WER
✓ Offline operation: $0 cost
✓ Speaker clustering: 2-6 auto-detected
✓ Flask API ready
✓ Multi-format: WAV, MP3, M4A, FLAC

**Current Limitations ⚠️**
⚠ Diarization: 60-70% (Target: 90%+)

**Root Cause:**
• Whisper chunks: 25s (too coarse)
• Sentence timing: estimated
• Guards use fake timing data

**Next Steps 🎯**

1. Enable word-level timestamps (2 weeks)
2. Use real word gaps (not estimates)
3. Test on diverse datasets
4. Target: 90%+ accuracy

**Innovation Report:**
✓ Submitted for review
⏳ Pyannote.audio API approval pending

---

DESIGN NOTES:

- 3-column layout: Working / Limitations / Next Steps
- Progress bar showing 70% complete
- Traffic light colors: green/yellow/red

---

================================================================================
SLIDE 8: FUZZY MATCHING - THE SITUATION
================================================================================

Title: From Webapp to Application-as-Recipe

**Existing Tool:**
✓ Working Fuzzy Matching webapp
✓ TF-IDF + Cosine Similarity
✓ Used by team

**Example Matches:**
"Microsoft Corporation" ↔ "Microsoft Corp"
"John Smith" ↔ "Jon Smith"
"555-1234" ↔ "(555) 1234"

**The Problem with Webapp:**
❌ Isolated from Dataiku workflow
→ Export data from Dataiku
→ Upload to separate webapp
→ Download results
→ Re-import to Dataiku

❌ Not reusable across projects
→ Copy-paste code each time
→ Inconsistent versions

❌ Manual steps break workflow

**The Requirement:**
🎯 Convert to Application-as-Recipe
✓ Keep inside Dataiku
✓ Make reusable (plug-and-play)
✓ No export/import needed

---

DESIGN NOTES:

- Split screen: Webapp (left) vs AAR (right)
- Show broken workflow with X marks
- Show smooth workflow with checkmarks

---

================================================================================
SLIDE 9: APPLICATION-AS-RECIPE FRAMEWORK
================================================================================

Title: What is Application-as-Recipe (AAR)?

**Definition:**
Reusable plugin that wraps Python code as Dataiku recipe

**AAR vs Regular Recipe:**

| Aspect      | Regular Recipe | Application-as-Recipe |
| ----------- | -------------- | --------------------- |
| Scope       | One project    | All projects          |
| Reuse       | Copy-paste     | Install once          |
| Updates     | Manual each    | Central update        |
| Consistency | Variable       | Standardized          |

**3 Core Components:**

**1. plugin.json**
• Plugin metadata
• Version, author, description

**2. recipe.json**
• Input/output roles
• Variable definitions
• Configuration schema

**3. recipe.py**
• Core processing logic
• Uses recipe.get_input()
• Uses recipe.get_output()

**Folder Structure:**

```
plugin-name/
├── plugin.json
├── custom-recipes/
│   └── recipe-name/
│       ├── recipe.json
│       └── recipe.py
└── python-lib/
    └── utils.py
```

---

DESIGN NOTES:

- Comparison table highlighted
- 3 component boxes with icons
- Folder tree visual

---

================================================================================
SLIDE 10: AAR CONVERSION PROCESS
================================================================================

Title: How I Converted to AAR

**Step 1: Research & Learning 📚**
✓ Searched Dataiku Community docs
✓ Studied plugin.json structure
✓ Analyzed example recipes
✓ Reviewed manager's EDA tool

**Step 2: Getting Help 🤝**
Shubham's Guidance:
• Explained recipe.get_input() vs dataiku.Folder()
• Helped debug variable schema issues
• Reviewed code structure
• Walked through plugin installation

**Step 3: Code Restructuring**

**Old (Webapp):**

```python
@app.route('/upload')
def upload():
    file1 = request.files['file1']
    # Process...
    return send_file(output)
```

**New (AAR):**

```python
input1 = dataiku.Dataset(
    get_input_names_for_role('input1')[0]
)
config = get_recipe_config()
output = dataiku.Dataset(
    get_output_names_for_role('output')[0]
)
```

**Key Changes:**
❌ Removed: Flask routes, file uploads
✅ Added: Recipe API, dynamic I/O
✅ Kept: Core TF-IDF logic unchanged

---

DESIGN NOTES:

- Code comparison side-by-side
- Highlight Shubham's help in callout box
- Show transformation arrows

---

================================================================================
SLIDE 11: AAR CHALLENGES SOLVED
================================================================================

Title: Technical Challenges & Solutions

**Challenge 1: Input/Output Roles**
❌ Problem: Didn't understand role-based access
✅ Solution: Use get_input_names_for_role('input1')
👤 Shubham: Explained role concept

**Challenge 2: Variable Schema**
❌ Problem: get_recipe_config() returned schema, not values
✅ Solution: Parse JSON to extract actual values
👤 Shubham: Provided helper function

**Challenge 3: Duplicate Output**
❌ Problem: Called dataiku.Folder() AND recipe API
✅ Solution: Use ONLY recipe API consistently
👤 Shubham: Identified conflict pattern

**Challenge 4: Plugin Installation**
❌ Problem: How to test plugin?
✅ Solution:

1.  Zip plugin folder
2.  Upload to Dataiku
3.  Install as plugin
4.  Add recipe to project
    👤 Shubham: Walked through process

**Best Practices Learned:**
✓ Follow manager's EDA pattern
✓ Use recipe API exclusively
✓ Define clear variable schemas
✓ Document everything

---

DESIGN NOTES:

- 4 challenge boxes in grid layout
- Icons for problem/solution
- Shubham's photo/icon with each help note

---

================================================================================
SLIDE 12: FUZZY MATCHING FINAL
================================================================================

Title: Final Implementation

**Architecture:**

```
Input Folder 1 ──┐
                 ├──> [Fuzzy Match Recipe] ──> Output
Input Folder 2 ──┘
                      ↑
              Project Variables (6 configs)
```

**6 Configurable Variables:**

1. threshold: 0-100 (default: 80)
2. columns_to_compare: ["name", "address"]
3. match_algorithm: "tfidf" or "exact"
4. output_format: "excel" or "csv"
5. max_results: Limit rows
6. include_scores: True/False

**Core Algorithm (Unchanged):**

1. Text Preprocessing
2. TF-IDF Vectorization
3. Cosine Similarity
4. Threshold Filtering
5. Output Generation

**Benefits:**
✓ Integrated into Dataiku workflows
✓ No export/import steps
✓ Reusable across projects
✓ Version controlled
✓ Consistent with team patterns

---

DESIGN NOTES:

- Architecture diagram as main visual
- Variable list in colored boxes
- Show before/after workflow comparison

---

================================================================================
SLIDE 13: TOPIC MINING CONVERSION
================================================================================

Title: From Streamlit App to AAR

**Existing Tool:**
✓ Streamlit app with LDA logic
✓ Running locally
✓ Working for team

**Example Analysis:**
"Platform is slow" → Performance Issues
"Love the UI design" → UI/UX Praise
"Support is helpful" → Support Quality

**The Problem with Streamlit:**
❌ Local-only (not integrated)
❌ Manual file copying
❌ Streamlit UI dependencies:
• st.file_uploader()
• st.dataframe()
• st.download_button()
❌ Tightly coupled code

**The Challenge:**
⚠️ First segregate UI from core logic

**The Solution:**

**Step 1: Code Segregation**
❌ Remove: All st.\* UI components
✅ Keep: Core LDA logic

**Step 2: Apply AAR Framework**
✓ Already knew AAR from Fuzzy Matching
✓ Main task: Clean separation

**Step 3: Shubham's Help**
• Explained segregation process
• Helped with column name issue
• Guided variable schema setup
• Reviewed at each stage

---

DESIGN NOTES:

- Split screen: Streamlit UI (left) vs Clean Recipe (right)
- Highlight what gets removed vs kept
- Show Shubham's involvement

---

================================================================================
SLIDE 14: TOPIC MINING FINAL
================================================================================

Title: Final AAR Implementation

**Architecture:**

```
Input Folder (survey_responses.xlsx)
         ↓
   [Topic Mining Recipe]
         ↓
Output Folder (survey_with_topics.xlsx)
         ↑
  Project Variables (3 configs)
```

**3 Configurable Variables:**

1. num_topics: 2-10 (default: 3)
2. text_column_name: "Answer" or "Comments"
3. chunk_size: 500 words (default)

**Core LDA Algorithm (Unchanged):**

1. Text Preprocessing
2. Document-Term Matrix
3. LDA Model Training (sklearn)
4. Topic Extraction
5. Output Generation

**Example Output:**
Topic 1 (32%): Performance Issues
Keywords: slow, crash, freeze, lag

Topic 2 (28%): UI/UX Praise
Keywords: design, interface, intuitive

Topic 3 (25%): Support Quality
Keywords: support, helpful, responsive

**Benefits:**
✓ Integrated into Dataiku
✓ No manual file copying
✓ Reusable across projects
✓ Column names configurable

---

DESIGN NOTES:

- Architecture diagram
- Example output in boxes
- Topic keyword clouds (optional)

---

================================================================================
SLIDE 15: SUMMARY & COLLABORATION
================================================================================

Title: Key Takeaways & Collaboration

**Project Summary:**

| Project        | Status     | Achievement           |
| -------------- | ---------- | --------------------- |
| Speech-to-Text | Prototype  | 5-12% WER, Rule-based |
| Fuzzy Matching | Production | Webapp → AAR          |
| Topic Mining   | Production | Streamlit → AAR       |

**Technical Achievements:**
🔹 Speech-to-Text: Rule-based diarization (4 guards)
🔹 Fuzzy Matching: AAR conversion complete
🔹 Topic Mining: Code segregation + AAR

**Key Learnings:**
✓ Webapp → Recipe: Replace Flask with recipe API
✓ Streamlit → Recipe: Remove UI, keep logic
✓ AAR Framework: Reusable plugin structure
✓ Variable handling: Parse JSON schemas
✓ Following patterns: Manager's EDA tool

**Collaboration Highlights:**
🤝 **Shubham:**
• Guided each conversion step
• Explained recipe API approach
• Debugged issues together
• Reviewed code structure

🤝 **Manager:**
• Provided EDA tool pattern
• Established best practices

🤝 **Team:**
• Topic mining collaboration
• Feedback and testing

**Next Steps:**
→ Speech-to-Text: Word-level timestamps (2 weeks)
→ Fuzzy Matching: Phonetic matching v2.0 (1 month)
→ Topic Mining: Pipeline integration (2 weeks)
→ Smart Scheduler: Documentation (6 days)

**Innovation Reports:**
✓ All three submitted
✓ Speech-to-Text under patent review

---

DESIGN NOTES:

- 3-column layout for summary
- Team collaboration icons/photos
- Timeline for next steps
- Q&A section at bottom

---

================================================================================
INSTRUCTIONS FOR GEMINI PPT GENERATION
================================================================================

**Prompt for Gemini:**

"Create a professional PowerPoint presentation with 15 slides using the content provided below.

Style Guidelines:

- Use a clean corporate template with blue/green color scheme
- Add relevant icons and graphics to each slide
- Use consistent fonts: Headers (32pt bold), Body (18pt)
- Include slide numbers
- Maintain white space for readability
- Use bullet points and short phrases (not paragraphs)
- Add transition effects between slides

For slides with **FLOWCHART PROMPT** sections:

- Generate the flowchart/diagram as described in the prompt
- Use professional colors and clean design
- Make diagrams easy to understand

For code sections:

- Use monospace font
- Light gray background
- Syntax highlighting if possible

Include these design elements:

- Title slide: Large title, subtitle, professional background
- Content slides: Title at top, 2-3 columns for content
- Comparison slides: Side-by-side layouts
- Summary slide: Table format with highlighted sections

[PASTE SLIDE CONTENT HERE]"

**For Flowcharts Only:**
If Gemini cannot generate flowcharts, use these separate prompts in:

- Lucidchart
- Draw.io
- Canva
- Microsoft Designer
- ChatGPT DALL-E

Then insert generated images into PowerPoint slides.

================================================================================
ADDITIONAL PROMPTS FOR IMAGE GENERATION
================================================================================

**Slide 4 - System Architecture Flowchart:**
"Create a professional vertical flowchart showing 4 stages of speech-to-text pipeline: (1) Audio Preprocessing with noise reduction, resampling, filtering in blue, (2) Whisper Transcription with chunking in green, (3) Speaker Diarization with MFCC and clustering in orange, (4) Output with timestamps in purple. Use rounded rectangles, arrows between stages, and icons for each component. Professional business style."

**Slide 6 - Decision Tree Flowchart:**
"Create a decision tree flowchart with diamond decision nodes and rectangular outcome boxes. Start with 'Sentence ends with ? . !' at top, then 4 decision diamonds asking about gap, continuation words, pronouns, and sentence length. Each YES path leads to green 'Keep Same Speaker' box, final NO leads to red 'Switch Speaker' box. Use arrows labeled YES/NO. Clean professional style."

**Slide 8 - Before/After Workflow:**
"Create a comparison diagram showing two workflows side by side. LEFT (red/orange): 'Dataiku → Export CSV → Webapp Upload → Process → Download → Import back to Dataiku' with broken workflow indicators. RIGHT (green/blue): 'Dataiku Input → Recipe → Output' smooth flow. Use icons for each step, arrows, and color coding to show improvement."

**Slide 12 - Architecture Diagram:**
"Create a simple architecture diagram showing two input folders converging into a central 'Fuzzy Match Recipe' box, then flowing to an output folder. Show project variables feeding into the recipe from below with an upward arrow. Use clean boxes, arrows, and minimal colors (blue for inputs, green for recipe, purple for output)."

**Slide 14 - Architecture Diagram:**
"Create a simple vertical flow diagram showing: Input Folder (survey data) → Topic Mining Recipe → Output Folder (with topics). Add a side arrow showing Project Variables (3 configs) feeding into the recipe. Use clean boxes, arrows, blue/green colors, and simple icons."

================================================================================
END OF DOCUMENT
================================================================================

SUMMARY:
✓ 15 slides total
✓ Concise, slide-ready content
✓ Flowchart prompts included for Slides 4 & 6
✓ Architecture diagram prompts for Slides 12 & 14
✓ Comparison visual prompts for Slide 8
✓ Ready to paste into Gemini for PPT generation
✓ You'll add screenshots/images separately
