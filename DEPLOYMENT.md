# 🚀 Deployment Guide - Diabetes Prediction ML

Complete guide for deploying your diabetes prediction system to various platforms.

---

## 📦 **Option 1: Docker Deployment (Recommended)**

### Prerequisites
- Docker installed ([Download](https://www.docker.com/products/docker-desktop))
- Docker Compose (included with Docker Desktop)

### Local Docker Deployment

1. **Build the Docker image:**
```bash
docker build -t diabetes-prediction-ml .
```

2. **Run the container:**
```bash
docker run -d -p 5000:5000 --name diabetes-ml diabetes-prediction-ml
```

3. **Access the app:**
```
http://localhost:5000
```

### Using Docker Compose

1. **Start the application:**
```bash
docker-compose up -d
```

2. **View logs:**
```bash
docker-compose logs -f
```

3. **Stop the application:**
```bash
docker-compose down
```

---

## ☁️ **Option 2: Deploy to Render**

### Step 1: Prepare for Render

1. **Create `render.yaml`** (already in project)

2. **Push to GitHub:**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

### Step 2: Deploy on Render

1. Go to [render.com](https://render.com)
2. Sign up / Log in
3. Click "New" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Name:** diabetes-prediction-ml
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python train_advanced.py`
   - **Start Command:** `python app.py`
   - **Plan:** Free

6. Click "Create Web Service"
7. Wait for deployment (5-10 minutes)
8. Access your app at: `https://diabetes-prediction-ml.onrender.com`

### Render Configuration File

```yaml
# render.yaml
services:
  - type: web
    name: diabetes-prediction-ml
    env: python
    buildCommand: pip install -r requirements.txt && python train_advanced.py
    startCommand: python app.py
    plan: free
    healthCheckPath: /
    envVars:
      - key: FLASK_ENV
        value: production
      - key: PYTHONUNBUFFERED
        value: 1
```

---

## 🚂 **Option 3: Deploy to Railway**

### Step 1: Prepare

1. **Create `railway.json`:**
```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "python app.py",
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 10
  }
}
```

### Step 2: Deploy

1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project" → "Deploy from GitHub repo"
4. Select your repository
5. Railway will auto-detect Python and deploy
6. Add environment variables:
   - `FLASK_ENV=production`
   - `PORT=5000`

7. Access at: `https://<your-project>.up.railway.app`

---

## ☁️ **Option 4: Deploy to AWS (EC2)**

### Step 1: Launch EC2 Instance

1. Go to AWS Console → EC2
2. Launch instance:
   - **AMI:** Ubuntu Server 22.04 LTS
   - **Instance type:** t2.micro (Free tier)
   - **Security Group:** Allow HTTP (80), HTTPS (443), Custom TCP (5000)

### Step 2: Connect and Setup

```bash
# Connect to instance
ssh -i your-key.pem ubuntu@<your-ec2-ip>

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3-pip python3-venv git -y

# Clone your repository
git clone <your-repo-url>
cd diabetes-prediction-ml

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset and train models
python download_dataset.py
python train_advanced.py

# Install PM2 for process management
sudo npm install -g pm2

# Start the app
pm2 start app.py --name diabetes-ml --interpreter python

# Make it run on startup
pm2 startup
pm2 save
```

### Step 3: Add Nginx (Optional)

```bash
sudo apt install nginx -y

# Configure Nginx
sudo nano /etc/nginx/sites-available/diabetes-ml

# Add:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/diabetes-ml /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## 🔒 **Option 5: Heroku Deployment**

### Step 1: Prepare Files

1. **Create `Procfile`:**
```
web: python app.py
```

2. **Create `runtime.txt`:**
```
python-3.10.12
```

### Step 2: Deploy

```bash
# Install Heroku CLI
# https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create diabetes-prediction-ml

# Set buildpack
heroku buildpacks:set heroku/python

# Deploy
git push heroku main

# Scale web dyno
heroku ps:scale web=1

# Open app
heroku open
```

---

## 🌐 **Environment Variables**

Set these environment variables for production:

```bash
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=<your-secret-key>
PORT=5000
PYTHONUNBUFFERED=1
```

---

## 📊 **Post-Deployment Checklist**

- [ ] Models are trained and saved
- [ ] All dependencies installed
- [ ] Environment variables set
- [ ] Health check endpoint working
- [ ] SSL certificate configured (for HTTPS)
- [ ] Domain name pointed (if using custom domain)
- [ ] Monitoring setup (optional)
- [ ] Backup strategy in place

---

## 🔍 **Monitoring & Logging**

### View Application Logs

**Docker:**
```bash
docker logs diabetes-ml -f
```

**Render/Railway:**
- Check dashboard logs section

**AWS:**
```bash
pm2 logs diabetes-ml
```

### Health Check

```bash
curl http://your-domain.com/
```

---

## 🛠️ **Troubleshooting**

### Common Issues

**Issue:** Port already in use  
**Solution:** Change port in `app.py` or use different port mapping

**Issue:** Models not found  
**Solution:** Ensure `train_advanced.py` runs during build/deployment

**Issue:** Out of memory  
**Solution:** Reduce model complexity or upgrade instance

**Issue:** Slow predictions  
**Solution:** Enable caching or use faster instance

---

## 💰 **Cost Estimates**

| Platform | Free Tier | Paid Plans |
|----------|-----------|------------|
| Render | 750 hrs/month | From $7/month |
| Railway | $5 credit/month | Pay as you go |
| Heroku | Discontinued | From $7/month |
| AWS EC2 | t2.micro (1 year) | From $3.80/month |
| Docker (Local) | Free | N/A |

---

## 🎯 **Best Practices**

1. **Use environment variables** for sensitive data
2. **Enable HTTPS** in production
3. **Set up monitoring** and alerts
4. **Implement rate limiting** to prevent abuse
5. **Regular backups** of models and data
6. **Update dependencies** regularly
7. **Use CDN** for static assets (optional)
8. **Implement logging** for debugging

---

## 📱 **Custom Domain Setup**

### For Render/Railway:

1. Go to Settings → Custom Domains
2. Add your domain: `diabetes.yourdomain.com`
3. Update DNS records:
   ```
   Type: CNAME
   Name: diabetes
   Value: <provided-by-platform>
   ```

### For AWS:

1. Get Elastic IP
2. Point your domain's A record to the IP
3. Configure SSL with Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

---

## 🚀 **Quick Deploy Commands**

### Render:
```bash
# Just push to GitHub, Render auto-deploys
git push origin main
```

### Railway:
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway up
```

### Docker:
```bash
docker-compose up -d
```

---

## 📞 **Support**

For deployment issues:
- Render: [render.com/docs](https://render.com/docs)
- Railway: [railway.app/help](https://railway.app/help)
- AWS: [aws.amazon.com/support](https://aws.amazon.com/support)

---

**Choose the deployment option that best fits your needs!**

**Recommended for beginners:** Render or Railway (easiest, free tier)  
**Recommended for production:** AWS or Docker (most control, scalable)

---

*Created by: Diabetes Prediction ML Team*  
*Last Updated: January 2026*
