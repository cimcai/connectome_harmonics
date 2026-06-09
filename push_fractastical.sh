#!/usr/bin/env bash
# Push to fractastical/connectome_harmonics for GitHub Pages.
# Prerequisite: create empty public repo at github.com/fractastical/connectome_harmonics
set -euo pipefail
cd "$(dirname "$0")"
git remote get-url fractastical >/dev/null 2>&1 || \
  git remote add fractastical https://github.com/fractastical/connectome_harmonics.git
git push -u fractastical main
echo "Done. Enable Pages: Settings → Pages → Source: GitHub Actions"
echo "Site: https://fractastical.github.io/connectome_harmonics/"
