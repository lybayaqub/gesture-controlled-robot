

# Gesture-Controlled Robot Simulation

An interactive, real-time dataset simulation environment that runs a trained computer vision model against the **LeapGestRecog** dataset to issue motion and speed commands to a virtual, top-down robot.

This project allows you to evaluate model accuracy, visualize path histories, and test robotic controls without needing a physical webcam or hardware.

---

## 🚀 Features

* **Webcam-Free Simulation:** Streams image sequences directly from the dataset directories to simulate real-time vision testing.


* **Dual-Panel HUD Canvas:**
* **Left Panel:** Displays the live dataset image, the model's resized 64×64 input thumbnail, live confidence tracking, and a real-time gesture-by-gesture accuracy breakdown.


* **Right Panel:** Features a top-down $10 \times 10$ coordinate grid arena tracking the robot's real-time position, directional heading arrow, and an 800-point path history trail.




* **Interactive Control System:** Instantly toggle between auto-cycle execution and manual frame stepping, modify playback delay intervals, or isolate specific hand gestures on the fly.



---

## 🛠️ Setup Instructions

### 1. Clone the Repository & Install Dependencies

Ensure you have Python installed, clone this repository, and install the required libraries:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
pip install numpy opencv-python tensorflow

```

### 2. Place Local Model Assets

Ensure your pre-trained weights and class configurations are placed directly in the root directory next to your script:

* `best_gesture_model.h5`

* `gesture_classes.npy`


### 3. Fetch the Dataset

Download and extract the **LeapGestRecog** dataset using the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download gti-upm/leapgestrecog --unzip -p .

```

Note: Ensure the extracted dataset folder is named `leapGestRecog` in your project root.

---

## 🎮 How to Run & Interface Controls

To launch the simulation windows, execute the main file:

```bash
python gesture_robot_live.py

```

### Keyboard Shortcuts Reference

Once the interface opens, use the following interactive key commands to control the simulation environment:

| Key | Action Summary | Description |
| --- | --- | --- |
| `SPACE` | **Next Image** | Steps forward to the next dataset frame manually (Manual Mode only).

 |
| `A` | **Toggle Auto / Manual** | Switches between automated looping and manual step-by-step frame testing.

 |
| `+` / `-` | **Adjust Cycle Speed** | Increases or decreases the automatic frame advancement delay interval.

 |
| `1` – `9`, `0` | **Jump to Gesture Class** | Instantly switches testing focus to an isolated gesture folder index (1–10).

 |
| `R` | **Reset Robot** | Resets the robot pose back to center grid coordinates `(5.0, 5.0)` with a 90° heading.

 |
| `C` | **Clear Trail** | Flushes the active position-tracking line logs from the canvas arena.

 |
| `Q` | **Quit Application** | Safely terminates the execution loops and prints a final accuracy report to the console.

 |

---

## 📊 Gesture Command Matrix

Predictions are mapped to robotic movement actions based on the following classification matrix:

| Index | Gesture Folder | Triggered Action | HUD UI Color Accent |
| --- | --- | --- | --- |
| **1** | `01_palm` | **STOP** | Soft Red

 |
| **2** | `02_l` | **TURN LEFT** | Pastel Blue

 |
| **3** | `03_fist` | **MOVE FORWARD** | Emerald Green

 |
| **4** | `04_fist_moved` | **MOVE BACKWARD** | Muted Orange

 |
| **5** | `05_thumb` | **SPEED UP** | Forest Green

 |
| **6** | `06_index` | **MOVE FORWARD** | Emerald Green

 |
| **7** | `07_ok` | **MOVE BACKWARD** | Muted Orange

 |
| **8** | `08_palm_moved` | **MOVE RIGHT** | Seafoam Green

 |
| **9** | `09_c` | **TURN LEFT** | Pastel Blue

 |
| **0** | `10_down` | **TURN RIGHT** | Soft Purple

 |
