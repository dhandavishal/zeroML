#!/bin/bash
# Script to remove large model files from git history and set up Git LFS

echo "🧹 Cleaning up large files from git history..."

# Remove the large files from git cache
echo "Removing large model files from git cache..."
git rm --cached models/oneclass_svm_optimized.pkl 2>/dev/null || true
git rm --cached models/*.pkl 2>/dev/null || true
git rm --cached models/*.keras 2>/dev/null || true
git rm --cached models/*.h5 2>/dev/null || true

# Create/update .gitignore to ignore model files
echo ""
echo "📝 Updating .gitignore..."
cat >> .gitignore << 'EOF'

# Model files (too large for GitHub)
models/*.pkl
models/*.keras
models/*.h5
models/*.pb
models/autoencoder_model/

# Keep only JSON result files
!models/*.json
EOF

# Stage the .gitignore changes
git add .gitignore

echo ""
echo "✅ Large model files removed from git tracking"
echo ""
echo "📋 Files that will be ignored:"
echo "   - models/*.pkl"
echo "   - models/*.keras"
echo "   - models/*.h5"
echo ""
echo "📋 Files that will be kept (results):"
echo "   - models/*.json"
echo ""
echo "💾 Next steps:"
echo "   1. Commit these changes:"
echo "      git commit -m 'Remove large model files and update .gitignore'"
echo ""
echo "   2. Push to GitHub:"
echo "      git push origin main"
echo ""
echo "📝 Note: Model files are now ignored and won't be pushed to GitHub."
echo "   You can regenerate them by running the training scripts."
