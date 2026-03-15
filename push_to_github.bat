@echo off
echo ==========================================
echo Pushing to GitHub Repository
echo ==========================================
echo.

REM Replace YOUR_USERNAME with your actual GitHub username
set /p USERNAME="Enter your GitHub username: "

echo.
echo Setting up remote repository...
git remote add origin https://github.com/%USERNAME%/silkworm-disease-predictor.git

echo.
echo Pushing code to GitHub...
git push -u origin master

echo.
echo ==========================================
echo Done! Now enable GitHub Pages:
echo 1. Go to https://github.com/%USERNAME%/silkworm-disease-predictor/settings/pages
echo 2. Source: Deploy from a branch
echo 3. Branch: master /docs folder
echo 4. Click Save
echo.
echo Your site will be live at:
echo https://%USERNAME%.github.io/silkworm-disease-predictor
echo ==========================================
pause
