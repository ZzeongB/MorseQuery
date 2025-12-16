# Deployment Instructions

## Option 1: Netlify (Easiest - Drag & Drop)

1. Go to https://app.netlify.com/drop
2. Create a folder with these files:
   - study_page1_collection.html
   - study_page2_labeling.html
   - OOPT_video.mp4 (your video file)
3. Drag the folder onto Netlify
4. Share the generated URL with participants

**Pros**: Zero setup, free, HTTPS automatic
**Cons**: 100MB file size limit (compress video if needed)

---

## Option 2: Vercel

```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd /Users/jeongin/morsequery
vercel --prod
```

Share the generated URL.

---

## Option 3: GitHub Pages

```bash
# Create new repo and push
git init
git add study_page1_collection.html study_page2_labeling.html OOPT_video.mp4
git commit -m "User study deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main

# Enable GitHub Pages in repo Settings > Pages > Deploy from main branch
```

Access at: `https://YOUR_USERNAME.github.io/YOUR_REPO/study_page1_collection.html`

---

## Option 4: Local Server (For In-Person Studies)

### Python (simplest)
```bash
cd /Users/jeongin/morsequery
python3 -m http.server 8000
```

Access at: `http://localhost:8000/study_page1_collection.html`

### Node.js
```bash
npx http-server -p 8000
```

---

## Option 5: Flask Backend (Recommended for Data Collection)

Instead of downloading JSON files from each participant's browser, collect data server-side.

Benefits:
- Centralized data storage
- No risk of participants forgetting to download data
- Automatic backup
- Can assign participant IDs

See `study_backend.py` for implementation.

---

## Video File Optimization

If video file is too large for static hosting:

```bash
# Compress video using ffmpeg
ffmpeg -i OOPT_video.mp4 -vcodec h264 -acodec aac -crf 28 OOPT_video_compressed.mp4
```

Or host video separately on:
- YouTube (unlisted): Update videoUrl to YouTube embed URL
- Vimeo (private)
- Cloud storage (Google Drive, Dropbox) with direct link

---

## Current Limitation: Client-Side Data Storage

⚠️ Current system stores data in sessionStorage (lost when tab closes).

For production studies, consider Option 5 (Flask backend) to persist data server-side.
