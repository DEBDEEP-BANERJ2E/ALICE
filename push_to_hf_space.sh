#!/bin/bash
# Push ALICE training files to HF Space

set -e

SPACE_URL="https://huggingface.co/spaces/rohanjain1648/alice-rl-training"
NGROK_URL="${1:-}"

if [ -z "$NGROK_URL" ]; then
    echo "Usage: ./push_to_hf_space.sh <NGROK_URL>"
    echo "Example: ./push_to_hf_space.sh https://abc123-def456.ngrok.io"
    exit 1
fi

echo "🚀 Pushing ALICE training to HF Space..."
echo "Space: $SPACE_URL"
echo "ngrok URL: $NGROK_URL"

# Clone Space
echo "📦 Cloning Space repo..."
rm -rf /tmp/alice-rl-training
git clone $SPACE_URL /tmp/alice-rl-training
cd /tmp/alice-rl-training

# Copy files
echo "📋 Copying files..."
cp ../ALICE/train.py .
cp ../ALICE/requirements.txt requirements-hf-spaces.txt
cp ../ALICE/Dockerfile.hf-spaces Dockerfile
cp ../ALICE/README_HF_SPACE.md README.md

# Update Dockerfile with ngrok URL
echo "🔧 Updating Dockerfile with ngrok URL..."
sed -i.bak "s|ENV ALICE_SERVER_URL=\"\"|ENV ALICE_SERVER_URL=\"$NGROK_URL\"|g" Dockerfile
rm -f Dockerfile.bak

# Commit and push
echo "📤 Committing and pushing..."
git add .
git commit -m "Add ALICE training with T4 GPU - $(date)"
git push

echo "✅ Done! Space is building..."
echo "Monitor at: $SPACE_URL"
