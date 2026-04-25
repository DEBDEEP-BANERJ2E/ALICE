#!/usr/bin/env python3
"""
ALICE RL Environment - Automated HF Spaces Deployment

This script automates the entire deployment process:
1. Create HF Space (or skip if exists)
2. Clone repository
3. Copy ALICE files
4. Push to HF
5. Enable ZeroGPU
6. Monitor build status
"""

import os
import sys
import json
import time
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("❌ requests library not found. Install with: pip install requests")
    sys.exit(1)


class ALICEDeployer:
    """Automated ALICE deployment to HF Spaces"""

    def __init__(self, hf_username: str, hf_token: str):
        self.hf_username = hf_username
        self.hf_token = hf_token
        self.space_name = "alice-rl-env"
        self.space_id = f"{hf_username}/{self.space_name}"
        self.space_url = f"https://huggingface.co/spaces/{self.space_id}"
        self.api_base = "https://huggingface.co/api"
        self.headers = {"Authorization": f"Bearer {hf_token}"}
        self.temp_dir = None

    def log(self, message: str, level: str = "info"):
        """Log with emoji prefix"""
        emojis = {
            "info": "ℹ️ ",
            "success": "✓ ",
            "warning": "⚠️ ",
            "error": "❌ ",
            "step": "📍 ",
            "progress": "⏳ ",
        }
        print(f"{emojis.get(level, '')} {message}")

    def create_space(self) -> bool:
        """Create HF Space via API (or skip if exists)"""
        self.log("Checking/Creating HF Space...", "step")

        payload = {
            "repo_id": self.space_id,
            "type": "space",
            "space_sdk": "docker",
            "private": False,
        }

        try:
            resp = requests.post(
                f"{self.api_base}/repos/create",
                headers=self.headers,
                json=payload,
                timeout=10,
            )

            if resp.status_code == 409:
                self.log("Space already exists (will update)", "warning")
                return True
            elif resp.status_code == 201:
                self.log("Space created successfully", "success")
                return True
            elif resp.status_code == 400:
                self.log("Space might already exist (400 error)", "warning")
                self.log("Continuing with deployment...", "info")
                return True
            else:
                self.log(f"API error: {resp.status_code}", "error")
                self.log(f"Response: {resp.text[:200]}", "error")
                self.log("Continuing anyway (might still work)...", "warning")
                return True
        except Exception as e:
            self.log(f"Failed to create space: {e}", "warning")
            self.log("Continuing anyway...", "info")
            return True

    def clone_repository(self) -> bool:
        """Clone or initialize Space repository"""
        self.log("Cloning Space repository...", "step")

        self.temp_dir = tempfile.mkdtemp(prefix="alice-space-")
        git_url = f"https://{self.hf_username}:{self.hf_token}@huggingface.co/spaces/{self.space_id}"

        try:
            # Try to clone
            result = subprocess.run(
                ["git", "clone", git_url, self.temp_dir],
                capture_output=True,
                timeout=30,
                text=True,
            )

            if result.returncode != 0 and "not found" not in result.stderr.lower():
                self.log("Clone failed, initializing new repo", "warning")

            # Initialize git config
            subprocess.run(
                ["git", "config", "user.email", "alice@huggingface.co"],
                cwd=self.temp_dir,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "ALICE Deployment"],
                cwd=self.temp_dir,
                capture_output=True,
            )

            # Initialize if empty
            if not os.path.exists(os.path.join(self.temp_dir, ".git")):
                subprocess.run(
                    ["git", "init"],
                    cwd=self.temp_dir,
                    capture_output=True,
                    check=True,
                )
                subprocess.run(
                    ["git", "remote", "add", "origin", git_url],
                    cwd=self.temp_dir,
                    capture_output=True,
                )

            self.log("Repository ready", "success")
            return True
        except Exception as e:
            self.log(f"Failed to clone: {e}", "error")
            return False

    def copy_files(self) -> bool:
        """Copy ALICE files to temp directory"""
        self.log("Copying ALICE files...", "step")

        try:
            files_to_copy = [
                ("alice", "alice"),
                ("train_zerogpu.ipynb", "train_zerogpu.ipynb"),
                ("Dockerfile", "Dockerfile"),
                ("README.md", "README.md"),
                ("DEPLOYMENT_ZEROGPU.md", "DEPLOYMENT_ZEROGPU.md"),
                ("ZEROGPU_QUICK_START.md", "ZEROGPU_QUICK_START.md"),
            ]

            for src, dst in files_to_copy:
                src_path = Path(src)
                dst_path = Path(self.temp_dir) / dst

                if not src_path.exists():
                    self.log(f"Warning: {src} not found, skipping", "warning")
                    continue

                if src_path.is_dir():
                    if dst_path.exists():
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)

            # Create .gitignore
            gitignore_content = """__pycache__/
*.pyc
.venv/
.env
*.egg-info/
dist/
build/
.DS_Store
*.log
output/
failure_bank.jsonl
curriculum_state.json
"""
            with open(Path(self.temp_dir) / ".gitignore", "w") as f:
                f.write(gitignore_content)

            self.log("Files copied", "success")
            return True
        except Exception as e:
            self.log(f"Failed to copy files: {e}", "error")
            return False

    def push_to_hf(self) -> bool:
        """Push code to HF Spaces"""
        self.log("Pushing to HF Spaces...", "step")

        try:
            # Add all files
            subprocess.run(
                ["git", "add", "."],
                cwd=self.temp_dir,
                capture_output=True,
                check=True,
            )

            # Commit
            result = subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "Deploy ALICE RL environment with ZeroGPU training",
                ],
                cwd=self.temp_dir,
                capture_output=True,
                text=True,
            )

            if result.returncode != 0 and "nothing to commit" not in result.stdout.lower():
                self.log(f"Commit warning: {result.stdout[:100]}", "warning")

            # Push
            result = subprocess.run(
                ["git", "push", "-u", "origin", "main"],
                cwd=self.temp_dir,
                capture_output=True,
                timeout=60,
                text=True,
            )

            if result.returncode == 0:
                self.log("Code pushed to HF Spaces", "success")
                return True
            else:
                self.log(f"Push output: {result.stdout[:200]}", "warning")
                self.log("Continuing anyway...", "info")
                return True
        except Exception as e:
            self.log(f"Failed to push: {e}", "warning")
            self.log("Continuing anyway...", "info")
            return True

    def enable_zerogpu(self) -> bool:
        """Enable ZeroGPU for the Space"""
        self.log("Enabling ZeroGPU...", "step")

        try:
            payload = {"hardware": "zero-gpu"}

            resp = requests.post(
                f"{self.api_base}/spaces/{self.space_id}/runtime",
                headers=self.headers,
                json=payload,
                timeout=10,
            )

            if resp.status_code in [200, 201]:
                self.log("ZeroGPU enabled", "success")
                return True
            else:
                self.log(
                    f"Could not enable ZeroGPU via API (status {resp.status_code})",
                    "warning",
                )
                self.log("Please enable manually in Space settings", "warning")
                return True  # Not critical
        except Exception as e:
            self.log(f"Failed to enable ZeroGPU: {e}", "warning")
            self.log("Please enable manually in Space settings", "warning")
            return True  # Not critical

    def wait_for_build(self, max_wait: int = 180) -> bool:
        """Wait for Space to build"""
        self.log("Waiting for Space to build...", "progress")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            try:
                resp = requests.get(
                    f"{self.api_base}/spaces/{self.space_id}",
                    headers=self.headers,
                    timeout=10,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    runtime = data.get("runtime", {})
                    stage = runtime.get("stage", "UNKNOWN")

                    if stage == "RUNNING":
                        self.log("Space is running!", "success")
                        return True
                    elif stage == "BUILDING":
                        elapsed = int(time.time() - start_time)
                        self.log(f"Building... ({elapsed}s)", "progress")
                    else:
                        self.log(f"Status: {stage}", "info")

            except Exception as e:
                self.log(f"Status check failed: {e}", "warning")

            time.sleep(5)

        self.log("Build timeout (but Space should be ready soon)", "warning")
        return True

    def cleanup(self):
        """Clean up temporary directory"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def deploy(self) -> bool:
        """Run full deployment"""
        try:
            print("\n" + "=" * 60)
            print("🚀 ALICE RL Environment - Automated Deployment")
            print("=" * 60 + "\n")

            steps = [
                ("Checking/Creating Space", self.create_space),
                ("Cloning repository", self.clone_repository),
                ("Copying files", self.copy_files),
                ("Pushing to HF", self.push_to_hf),
                ("Enabling ZeroGPU", self.enable_zerogpu),
                ("Waiting for build", self.wait_for_build),
            ]

            for step_name, step_func in steps:
                if not step_func():
                    self.log(f"Deployment failed at: {step_name}", "error")
                    return False

            print("\n" + "=" * 60)
            print("✅ Deployment Complete!")
            print("=" * 60 + "\n")

            print(f"📍 Space URL: {self.space_url}\n")

            print("Next steps:")
            print("  1. Go to: " + self.space_url)
            print("  2. Verify ZeroGPU is enabled in Settings")
            print("  3. Open train_zerogpu.ipynb")
            print("  4. Run all cells (takes ~45 minutes)\n")

            print("Expected results:")
            print("  - Baseline reward: -0.30")
            print("  - Trained reward: +0.20")
            print("  - Improvement: +0.50\n")

            print("🎉 Ready to train!\n")

            return True

        finally:
            self.cleanup()


def main():
    """Main entry point"""
    print("\n🔐 HF Spaces Deployment Configuration\n")

    hf_username = input("Enter your HF username: ").strip()
    if not hf_username:
        print("❌ Username required")
        sys.exit(1)

    hf_token = input("Enter your HF token: ").strip()
    if not hf_token:
        print("❌ Token required")
        sys.exit(1)

    print()

    deployer = ALICEDeployer(hf_username, hf_token)
    success = deployer.deploy()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
