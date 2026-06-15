# Gesture-Controlled Robot Simulation 🤖🖐️

An interactive, real-time dataset simulation that runs a trained computer vision model against the **LeapGestRecog** dataset to issue motion and speed commands to a virtual top-down robot. Built with Python, OpenCV, and TensorFlow.

## 🚀 Features
* **Dual-Panel HUD:** Features a live inference analysis panel on the left (showing true vs. predicted labels, live accuracy tracking, and model confidence) paired with a live top-down grid arena tracking the robot's coordinates, heading vector, and path history trail on the right.
* **Dataset Injection Mode:** Simulates running a live camera feed without requiring a physical webcam by streaming image sequences from target gesture directories.
* **Interactive Control System:** Instantly toggle between manual frame stepping and auto-cycle execution modes, fine-tune playback delay on the fly, or quickly jump across target hand gestures using keyboard shortcuts.

---

## 🛠️ Setup Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/gesture-robot-simulation.git](https://github.com/YOUR_USERNAME/gesture-robot-simulation.git)
cd gesture-robot-simulation
2. Install DependenciesEnsure you have Python 3.8+ installed, then run:Bashpip install -r requirements.txt
3. Acquire Assets & DatasetPlace your trained model assets (best_gesture_model.h5 and gesture_classes.npy) into the root directory of this project.Download and extract the LeapGestRecog dataset into a folder named leapGestRecog/ using the Kaggle API:Bashpip install kaggle
kaggle datasets download gti-upm/leapgestrecog --unzip -p .
🎮 How to Run & ControlsExecute the primary script to launch the interface environment:Bashpython gesture_robot_live.py
Keyboard ShortcutsKeyActionSPACEStep to the next image (Manual Mode only)AToggle Auto-advance Mode / Manual Mode+ / -Speed up / Slow down automatic cycle interval1 – 9, 0Instantly swap evaluation focus to a specific gesture classRReset robot pose back to center home (5.0, 5.0)CFlush position history line trail logs from the canvas arenaQSafely terminate application loops and display final evaluation metrics📊 Gesture Mapping ReferenceThe system interprets your model predictions using the following logic matrix:Folder IndexTarget ClassBound ActionHUD Theme101_palmSTOPSoft Red202_lTURN LEFTPastel Blue303_fistMOVE FORWARDEmerald Green404_fist_movedMOVE BACKWARDMuted Orange505_thumbSPEED UPForest Green606_indexMOVE FORWARDEmerald Green707_okMOVE BACKWARDMuted Orange808_palm_movedMOVE RIGHTSeafoam Green909_cTURN LEFTPastel Blue010_downTURN RIGHTSoft Purple
---

## 4. Git Commands to Push to GitHub

Once you have created a new, blank public repository on your GitHub account, open your local terminal inside your project directory and run these commands to initialize and upload it:

```bash
# Initialize local directory as a Git repository
git init

# Stage all files except those listed in your .gitignore
git add .

# Commit your files locally
git commit -m "Initial commit: Add interactive robot simulation script and setup configuration docs"

# Point your local repository to your remote GitHub page
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Set your main branch name to main
git branch -M main

# Push your changes to the cloud
git push -u origin main
