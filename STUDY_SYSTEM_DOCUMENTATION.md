# User Study System: Intent Collection During Auditory Content Consumption

## System Overview

A two-page web-based system for collecting time-aligned user intent data during video watching through micro-gestures and retrospective labeling.

---

## Page 1: Data Collection (`study_page1_collection.html`)

### UI Layout
```
┌────────────────────────────────────────────────────┐
│  Press any button whenever you want the system to  │
│  do something for you.                             │
│                                                     │
│  There is no right or wrong timing.                │
├────────────────────────────────────────────────────┤
│                                                     │
│              [   Video Player   ]                  │
│                                                     │
│                                                     │
│                                    [Next →]        │
└────────────────────────────────────────────────────┘
```

### Interaction Flow
1. Participant watches video
2. Participant presses any keyboard button when they want system assistance
3. Each button press logs: `{ gesture_timestamp, video_time }`
4. No visual feedback, no interruption
5. When complete, participant clicks "Next →" to proceed to Page 2

### Data Captured
- `gesture_timestamp`: ISO 8601 timestamp of button press
- `video_time`: Current video playback position (seconds)

---

## Page 2: Retrospective Labeling (`study_page2_labeling.html`)

### UI Layout (per gesture)
```
┌────────────────────────────────────────────────────┐
│  Gesture 1 of 5                                    │
├────────────────────────────────────────────────────┤
│              [   Video Player   ]                  │
│           (±30 sec around gesture)                 │
├────────────────────────────────────────────────────┤
│  Transcript for this window:                       │
│  ... text text text HIGHLIGHTED text text ...      │
├────────────────────────────────────────────────────┤
│  Q1. Intended Action (select one)                  │
│  ○ Look up an unfamiliar word or concept           │
│  ○ Translate a phrase                              │
│  ○ Clarify an ambiguous or similar-sounding term   │
│  ○ Replay or review a missed segment               │
│  ○ Bookmark a segment for later                    │
│  ○ Trigger a follow-up action                      │
│  ○ Other [____________]                            │
├────────────────────────────────────────────────────┤
│  Q2. Target of Action                              │
│  ○ A specific word or phrase [____________]        │
│  ○ The broader context [text area]                 │
│  ○ Implicit or assumed background knowledge        │
│    [text area]                                     │
│  ○ Other [____________]                            │
├────────────────────────────────────────────────────┤
│  Q3. Desired Output Form                           │
│  ○ Text-based search results                       │
│  ○ Image-based results                             │
│  ○ Text + image results                            │
│  ○ Short synthesized summary                       │
│  ○ Structured format (table, timeline, list)       │
│  ○ Audio response                                  │
│  ○ Actionable output (definition, link, task)      │
│  ○ Other [____________]                            │
├────────────────────────────────────────────────────┤
│                                      [Submit]      │
└────────────────────────────────────────────────────┘
```

### Interaction Flow
1. System loads first gesture
2. Video automatically cued to gesture_time - 30 seconds
3. Transcript displays 1-minute window with gesture timestamp highlighted
4. Participant answers Q1, Q2, Q3
5. Participant clicks "Submit"
6. System loads next gesture
7. Repeat until all gestures labeled
8. On completion: labeled data automatically downloads as JSON

---

## Data Schema

### Output Format
```json
[
  {
    "gesture_timestamp": "2025-12-16T14:23:45.123Z",
    "video_time": 127.45,
    "intended_action": "lookup",
    "intended_action_other": null,
    "target_type": "specific_word",
    "selected_text": "quantum entanglement",
    "target_description": null,
    "target_other": null,
    "desired_output_form": "text_image_results",
    "desired_output_other": null
  },
  {
    "gesture_timestamp": "2025-12-16T14:25:12.456Z",
    "video_time": 215.83,
    "intended_action": "other",
    "intended_action_other": "Set a reminder to research this topic",
    "target_type": "broader_context",
    "selected_text": null,
    "target_description": "The entire explanation about climate feedback loops",
    "target_other": null,
    "desired_output_form": "actionable",
    "desired_output_other": null
  }
]
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `gesture_timestamp` | ISO 8601 string | Exact time button was pressed |
| `video_time` | float | Video playback position in seconds |
| `intended_action` | enum | One of: lookup, translate, clarify, replay, bookmark, followup, other |
| `intended_action_other` | string/null | Free text if "other" selected for Q1 |
| `target_type` | enum | One of: specific_word, broader_context, background_knowledge, other |
| `selected_text` | string/null | Text selected from transcript (if specific_word) |
| `target_description` | string/null | Free text description (if broader_context or background_knowledge) |
| `target_other` | string/null | Free text if "other" selected for Q2 |
| `desired_output_form` | enum | One of: text_results, image_results, text_image_results, summary, structured, audio, actionable, other |
| `desired_output_other` | string/null | Free text if "other" selected for Q3 |

---

## Study Setup Instructions

### Before Each Session

1. **Prepare video file**: Place video in same directory as HTML files
2. **Set video URL**: In browser console before Page 1:
   ```javascript
   sessionStorage.setItem('videoUrl', 'your_video.mp4');
   ```

3. **Prepare transcript data** (optional): In browser console before Page 2:
   ```javascript
   sessionStorage.setItem('transcriptData', JSON.stringify({
     "0.0": "Welcome to this lecture on",
     "2.1": "quantum mechanics. Today we'll",
     "5.3": "explore entanglement and",
     // ... time-aligned transcript segments
   }));
   ```

### During Session

1. Participant opens `study_page1_collection.html`
2. Participant watches and presses buttons
3. Participant clicks "Next →"
4. Participant labels all gestures on Page 2
5. Labeled data auto-downloads as JSON file

### Data Collection

- Data stored in browser sessionStorage during study
- Final labeled data downloads automatically as `labeled_gestures_[timestamp].json`
- Each participant generates one JSON file per session

---

## Scenarios for Testing

### Scenario 1: Lecture Video (Cognitively Demanding)
- **Example**: University physics lecture on quantum mechanics
- **Expected behaviors**: Frequent lookups, clarifications, bookmarks
- **Typical gestures**: 5-15 per 10-minute segment

### Scenario 2: Casual TV Conversation (Easy, Non-demanding)
- **Example**: Talk show interview or sitcom
- **Expected behaviors**: Occasional translation, replay
- **Typical gestures**: 1-5 per 10-minute segment

---

## Technical Notes

- Uses browser sessionStorage (data cleared when tab closes)
- No backend required
- No external dependencies
- Works offline once files are loaded
- Tested in modern Chrome/Firefox/Safari
