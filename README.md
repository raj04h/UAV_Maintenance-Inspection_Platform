# UAV Maintenance App

An Android application designed to streamline UAV (Unmanned Aerial Vehicle) maintenance and inspection workflows. 
The app leverages machine learning (TensorFlow Lite) for damage detection, generative AI for technical solutions, QR/RFID scanning for asset identification, and Firebase for real-time data storage.

# App Working
![image](https://github.com/user-attachments/assets/8f31ac73-7625-4177-afd9-c7a79f674ccd)

# Features
Uses a TensorFlow Lite model (UavDamage) to detect various damage types (scratches, pitting, etc.).
Captures images via device camera.
QR/RFID Scanner

Leverages Google ML Kit’s BarcodeScanning API to read QR codes or RFID tags.
Automatically identifies the UAV or part for easier logging.
Technical Error Handling.

Categorizes common UAV errors (GPS, Power, Software).
Stores errors in Firebase Realtime Database for tracking and analytics.
AI-driven Solutions.

Integrates a generative AI model (Gemini) to provide potential solutions or troubleshooting steps based on detected errors/damage.
Live UAV Location.

Displays real-time UAV location (placeholder or actual GPS tracking).
Allows for quick reference of UAV position during maintenance or inspection.
Maintenance Checklists

Physical and Technical checklists to ensure thorough inspections.
Timestamps and logs saved for future reference.
Firebase Integration

Stores error logs, maintenance data, and scanned results in a Realtime Database.

Can be extended to sync across multiple devices or technicians.

# Tech Stack🚡
Language: Kotlin, Python.

IDE: Android Studio, VS Code.

ML: TensorFlow Lite for damage detection.

ML: Prophet Model for Next date for service.

AI: Gemini AI (GenerativeModel) for generating troubleshooting solutions.

Barcode/QR: Google ML Kit.

Database: Firebase Realtime Database.

UI: Jetpack ViewBinding, Android XML Layouts.
