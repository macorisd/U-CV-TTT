# Ultimate Computer Vision TicTacToe (U-CV-TTT)

## Overview
**Ultimate Computer Vision TicTacToe** is a Python project that simulates a TicTacToe player whose "eyes" are the camera of a mobile device. The application uses Python and OpenCV to recognize and interact with the TicTacToe board.

## Key Features
- Play against an AI with three difficulty modes: Easy, Medium, and Nightmare.
- Real-time board recognition via a mobile device camera.
- Train and test your own symbol classifier for custom configurations.

## Requirements
- Python (latest version recommended)
- DroidCam installed on your mobile device

## Installation
Follow these steps to set up the project:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/macorisd/U-CV-TTT.git
   ```

2. **Install Python**
   - Go to [python.org/downloads](https://www.python.org/downloads/).
   - Download and install the latest version of Python.

3. **Install DroidCam on a Mobile Device**
   - For Android: [Download from Google Play Store](https://play.google.com/store/apps/details?id=com.dev47apps.droidcam&hl=es&gl=US&pli=1).

4. **Install Required Python Packages**
   - Open the project folder, named `U-CV-TTT`.
   - Right-click in the folder and select **Open in Terminal**.
   - Install dependencies using one of the following methods:

     **Option 1:** Run the batch file `install_packages.bat` located in the `Setup` folder.

     **Option 2:** Install packages manually by running the following commands:
     ```bash
     pip install opencv-python
     pip install shapely
     pip install matplotlib
     ```

5. **You're ready to play!**
   - Open the file `U-CV-TTT.pyw` for direct execution, or run `U-CV-TTT.py` to view the Python code.

## How to Play

1. Connect both your mobile device and computer to the same Wi-Fi network (or tether your computer to your mobile data).
2. Launch DroidCam on your mobile device.
3. Run the `U-CV-TTT.pyw` file.
4. Click on the **IP/Port** button in the application window.
5. Enter the IP and port provided by DroidCam into the corresponding fields.
6. Select a difficulty mode: Easy, Medium, or Nightmare.
7. Capture board images by pressing the spacebar. Exit the application by pressing `q`.

## Tips for Optimal Performance
- Use plain white paper and a black marker for better recognition.
- Align the TicTacToe board approximately within the provided grid, but avoid placing it too close to the camera.
- You may choose to draw or not draw the bot's moves on the paper; both approaches work.

## Advanced Features

1. **Train Your Own Symbol Classifier**
   - Use the `train_classifier.py` script to train a custom symbol classifier.

2. **Test the Symbol Classifier**
   - Use the `test_classifier.py` script to evaluate your classifier's performance.

## Author
Made with ❤️ by **Macorís Decena Giménez**.  
[GitHub Profile](https://github.com/macorisd)

## Reporting Issues
If you encounter any bugs or issues, please feel free to contact me at **macorisd@gmail.com**.