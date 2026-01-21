# 📱 React Frontend for Diabetes Prediction ML

Modern React application for the diabetes prediction system.

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ and npm
- Backend API running on `http://localhost:5000`

### Installation

```bash
cd react-frontend
npm install
npm start
```

The app will open at `http://localhost:3000`

## 📁 Project Structure

```
react-frontend/
├── public/
│   ├── index.html
│   └── manifest.json
├── src/
│   ├── components/
│   │   ├── PredictionForm.jsx
│   │   ├── Results.jsx
│   │   ├── ModelMetrics.jsx
│   │   └── Header.jsx
│   ├── services/
│   │   └── api.js
│   ├── styles/
│   │   └── App.css
│   ├── App.jsx
│   └── index.js
├── package.json
└── README.md
```

## 🎨 Features

- ✅ Modern React 18 with Hooks
- ✅ Material-UI components
- ✅ Real-time predictions
- ✅ Beautiful visualizations with Chart.js
- ✅ Responsive design
- ✅ Dark/Light theme toggle
- ✅ Form validation
- ✅ Loading states
- ✅ Error handling

## 🔧 Configuration

Update API endpoint in `src/services/api.js`:

```javascript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';
```

## 📦 Build for Production

```bash
npm run build
```

The optimized files will be in the `build/` directory.

## 🌐 Deploy React App

### Option 1: Vercel
```bash
npm install -g vercel
vercel
```

### Option 2: Netlify
```bash
npm install -g netlify-cli
netlify deploy --prod
```

### Option 3: GitHub Pages
```bash
npm install --save gh-pages

# Add to package.json:
"homepage": "https://yourusername.github.io/diabetes-prediction",
"predeploy": "npm run build",
"deploy": "gh-pages -d build"

# Deploy
npm run deploy
```

## 🔌 API Integration

The React app connects to your Flask backend:

- **Prediction:** `POST /predict`
- **Metrics:** `GET /api/metrics`

## 💻 Development

```bash
# Start development server
npm start

# Run tests
npm test

# Build for production
npm run build

# Lint code
npm run lint
```

## 🎯 Environment Variables

Create `.env` file:

```
REACT_APP_API_URL=http://localhost:5000
REACT_APP_TITLE=Diabetes Prediction AI
```

## 📱 Mobile App (React Native)

To convert to React Native:

1. Use React Native CLI or Expo
2. Replace Material-UI with React Native Paper
3. Update navigation with React Navigation
4. Use AsyncStorage for local data

See `REACT_NATIVE.md` for detailed guide.

---

**Created with React 18 and ❤️**
