# Railway Deployment Guide

## Current Issue

Your frontend is trying to connect to:

```
https://flowiqintelligenttrafficanalytics-production.up.railway.app
```

But this domain cannot be resolved, which means the backend isn't deployed or the URL is incorrect.

## Solution Options

### Option 1: Deploy to Railway (Recommended)

#### Using Railway Dashboard:

1. Go to [railway.app](https://railway.app)
2. Create a new project or select existing project
3. Connect your GitHub repository
4. Railway will auto-detect your configuration from `railway.toml`
5. Add environment variables if needed
6. Deploy!
7. **Copy the actual deployment URL** from Railway dashboard
8. Update `platform/frontend/js/app.js` line 13 with the correct URL

#### Using Railway CLI:

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Link to your project
railway link

# Deploy
railway up

# Get the deployment URL
railway domain
```

### Option 2: Run Backend Locally

If you want to test locally first:

```bash
# Install dependencies
cd /home/user/Desktop/traffic_analysis
pip3 install -r platform/requirements.txt

# Start backend
cd platform/backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then update `platform/frontend/js/app.js` line 11-13 to:

```javascript
const API_BASE = "/api"; // Use local backend
```

And serve the frontend:

```bash
cd platform/frontend
python3 -m http.server 3000
```

Visit: http://localhost:3000

### Option 3: Check Railway Logs

If you believe it's already deployed:

1. Go to Railway dashboard
2. Click on your service
3. Check "Deployments" tab for errors
4. Check "Logs" tab for runtime errors
5. Verify the correct domain in "Settings" → "Domains"

## Common Issues

### Issue 1: Wrong Domain

**Symptom**: DNS cannot resolve
**Fix**: Copy the exact domain from Railway dashboard

### Issue 2: Build Failed

**Symptom**: No deployment URL available
**Fix**: Check Railway logs, ensure all dependencies are in `requirements.txt`

### Issue 3: Port Configuration

**Symptom**: Service starts but doesn't respond
**Fix**: Ensure `railway.toml` uses `$PORT` environment variable:

```toml
[tool.railway]
startCommand = "cd platform/backend && uvicorn main:app --host 0.0.0.0 --port $PORT"
```

### Issue 4: CORS Still Blocking

**Symptom**: Backend responds but CORS error persists
**Fix**: Already updated in `main.py` - redeploy to Railway

## Next Steps

1. **Verify Railway deployment status**
2. **Get the correct deployment URL**
3. **Update frontend API_BASE URL**
4. **Test the connection**

## Testing Your Backend

Once deployed, test with:

```bash
curl https://YOUR-ACTUAL-RAILWAY-URL.railway.app/api/health
```

Should return:

```json
{ "status": "healthy", "timestamp": "..." }
```
