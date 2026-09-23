#!/bin/bash
# Deploy both tools to GitHub Pages
# Run this after creating the two GitHub repositories and a Personal Access Token (PAT)

set -e

GITHUB_USER="nidhisukhija"
DISEASE_REPO="silkworm-disease-predictor"
MOTH_REPO="maada-shalabh-parikshan"

echo "========================================"
echo "Deploying both tools to GitHub Pages"
echo "========================================"
echo ""
echo "Prerequisites:"
echo "1. Create these empty repos on GitHub (NO README, NO .gitignore):"
echo "   - https://github.com/$GITHUB_USER/$DISEASE_REPO"
echo "   - https://github.com/$GITHUB_USER/$MOTH_REPO"
echo "2. Create a GitHub Personal Access Token (classic) with 'repo' scope:"
echo "   https://github.com/settings/tokens/new"
echo "3. When prompted below, enter your GitHub username and the token as password."
echo ""

# Deploy Disease Predictor
echo "--- Deploying Silkworm Disease Predictor ---"
cd "/Volumes/NO NAME/0000_Project/Nidhi_RC/DMR"
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USER/$DISEASE_REPO.git"
git push -u origin main

echo ""
echo "--- Deploying Mother Moth Examination Guide ---"
cd "/Volumes/NO NAME/0000_Project/Nidhi_RC/DMR-deploy-maada-shalabh-parikshan"
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USER/$MOTH_REPO.git"
git push -u origin main

echo ""
echo "========================================"
echo "Pushes complete. Now enable GitHub Pages:"
echo "========================================"
echo "1. Open https://github.com/$GITHUB_USER/$DISEASE_REPO/settings/pages"
echo "   Source: Deploy from a branch"
echo "   Branch: main  ->  /docs"
echo "   Save. URL will be: https://$GITHUB_USER.github.io/$DISEASE_REPO/"
echo ""
echo "2. Open https://github.com/$GITHUB_USER/$MOTH_REPO/settings/pages"
echo "   Source: Deploy from a branch"
echo "   Branch: main  ->  / (root)"
echo "   Save. URL will be: https://$GITHUB_USER.github.io/$MOTH_REPO/"
echo ""
echo "Wait 1-2 minutes, then open the URLs above."
