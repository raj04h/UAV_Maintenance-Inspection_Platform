# UAV-MRO: AI-Powered UAV Maintenance & Inspection Platform

An end-to-end AI-powered UAV Maintenance, Repair and Overhaul (MRO) platform that combines an Android application, Computer Vision, Machine Learning, Predictive Analytics, and Generative AI to automate UAV inspection, maintenance workflows, fleet lifecycle management, and landing-site safety analysis.

The platform enables technicians to inspect UAVs using a mobile application while an AI backend performs automated visual inspection, predictive maintenance, and intelligent decision support.

---

# System Overview

```text
                     UAV Landing
                          │
                          ▼
               QR / RFID Identification
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
 Android Inspection App              CCTV AI Inspection
        │                                   │
        ▼                                   ▼
Image Capture                    Drone Body Detection
        │                                   │
        ▼                                   ▼
TensorFlow Lite Damage Model      Defect Detection Model
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
             Firebase Realtime Database
                       │
                       ▼
        Maintenance History & Lifecycle
                       │
                       ▼
     Predictive Maintenance (Prophet Model)
                       │
                       ▼
        Gemini AI Technical Recommendations
                       │
                       ▼
        Maintenance Dashboard & Reports
```

---

# Mobile Application

The Android application streamlines UAV inspection and maintenance activities performed by technicians in the field.

## App Workflow

![image](https://github.com/user-attachments/assets/8f31ac73-7625-4177-afd9-c7a79f674ccd)

---

# AI Backend

The backend performs automated UAV inspection using Computer Vision and Machine Learning.

## Model Testing Sample

<img width="986" height="715" alt="image" src="https://github.com/user-attachments/assets/9116bb48-9e4e-4218-a4dd-2daebae7946d" />

---

## Stage 1 — UAV Body Detection

<img width="1953" height="765" alt="image" src="https://github.com/user-attachments/assets/5587230b-49af-4538-a2b7-3383cb3420d9" />

---

## Stage 2 — UAV Defect Detection

<img width="1736" height="867" alt="image" src="https://github.com/user-attachments/assets/1493978b-793c-4fb3-9555-02d7b27a3b92" />

---

# Key Features

## Android Application

### UAV Damage Detection

- Captures UAV images using the device camera
- Runs TensorFlow Lite models on-device
- Detects defects including:
  - Scratches
  - Paint Damage
  - Corrosion
  - Pitting
  - Structural Damage

---

### QR / RFID Asset Identification

- Google ML Kit Barcode Scanner
- QR Code Detection
- RFID Support
- Automatically identifies:
  - UAV
  - Components
  - Maintenance History

---

### Maintenance Checklists

Supports structured maintenance procedures including

- Physical Inspection
- Technical Inspection
- Component Validation
- Timestamp Logging
- Inspection Reports

---

### Technical Error Management

Categorizes common UAV faults including

- GPS Issues
- Power System Errors
- Software Faults
- Sensor Failures

All maintenance logs are stored in Firebase Realtime Database.

---

### AI Technical Assistant

Uses Gemini AI to generate

- Root Cause Analysis
- Repair Suggestions
- Troubleshooting Steps
- Maintenance Recommendations

---

### Live UAV Tracking

Displays

- Current UAV Location
- Fleet Status
- Inspection Location

Can be extended for real-time fleet monitoring.

---

### Firebase Integration

Stores

- Damage Reports
- Inspection Logs
- Technician Reports
- Maintenance History
- QR Scan Results

Supports real-time synchronization across multiple devices.

---

# AI Backend Features

## Automated UAV Analysis

### QR Detection

- Detects UAV identity using QR markers
- Retrieves UAV maintenance history
- Associates inspections with the correct aircraft

---

### UAV Body Detection

Automatically detects

- Drone Body
- Frame Position
- Inspection Region

before running defect detection.

---

### Defect Detection

Computer Vision models detect

- Paint Damage
- Cracks
- Corrosion
- Surface Wear
- Pitting
- Structural Defects

Generates automated maintenance alerts.

---

### Assembly Validation

Detects installation issues such as

- Incorrect Propeller Direction
- Missing Components
- Assembly Faults

Reduces technician errors during UAV assembly.

---

### Annotation Pipeline

Custom datasets include

- Drone Body Parts
- Component Labels
- Damage Classes
- Defect Segmentation

---

# Flight Lifecycle Analytics

## Predictive Maintenance

Uses Prophet forecasting to estimate

- Next Maintenance Date
- Expected Service Window
- Repair Schedule

based on

- Flight Hours
- Historical Repairs
- Previous Defects

---

## Lifecycle Monitoring

Tracks

- UAV Health
- Component Wear
- Maintenance Frequency
- Damage Progression

Generates lifecycle graphs for long-term fleet monitoring.

---

## Quality Benchmarking

Monitors

- Paint Wear
- Corrosion Progression
- Environmental Damage
- Structural Aging

---

# Smart Landing Site Management

Automatically evaluates landing zones.

### Landing Site Detection

- Detects available landing pads
- Verifies landing clearance

---

### Safety Analysis

Checks

- Obstacles
- Surface Safety
- Landing Availability

---

### Alternate Landing Suggestion

If landing is unsafe

- Finds alternate coordinates
- Suggests safer landing locations

---

# Engineering Objectives

- AI-driven UAV Inspection
- Automated Defect Detection
- Predictive Maintenance
- Fleet Lifecycle Management
- Smart Landing Safety
- Technician Decision Support
- AI-powered Repair Recommendations
- Digital Maintenance Records

---

# Tech Stack

## Mobile

- Kotlin
- Android Studio
- Android XML
- ViewBinding
- Firebase Realtime Database
- Google ML Kit

---

## Artificial Intelligence

- TensorFlow Lite
- Computer Vision
- Object Detection
- Defect Detection
- QR Detection
- Prophet Forecasting
- Gemini AI

---

## Backend

- Python
- OpenCV
- NumPy
- Pandas

---

## Database

- Firebase Realtime Database

---

## Dashboard

- HTML
- CSS

---

# Project Architecture

<img width="1648" height="959" alt="image" src="https://github.com/user-attachments/assets/3fdb3024-f45a-44f2-9395-d72d3177e646" />

---

## Backend Structure

<img width="379" height="183" alt="image" src="https://github.com/user-attachments/assets/d156002a-0497-4731-8abf-94c9f1f7be8d" />

---

# Project Structure

```text
UAV-MRO/
│
├── Android-App/
│   ├── Camera Module
│   ├── QR Scanner
│   ├── Damage Detection
│   ├── Firebase Integration
│   ├── Gemini AI
│   └── Maintenance Checklist
│
├── Backend/
│   ├── main.py
│   ├── visual_feed.py
│   ├── weathers.py
│   ├── Prophet Prediction
│   ├── Defect Detection
│   ├── QR Detection
│   └── Landing Safety
│
├── Models/
│
├── Dashboard/
│   └── info_PJ1.html
│
├── Dataset/
│
└── README.md
```

---

# Backend Components

| File | Description |
|------|-------------|
| main.py | Core UAV maintenance logic |
| visual_feed.py | CCTV processing and defect detection |
| weathers.py | Environmental condition monitoring |
| info_PJ1.html | UAV lifecycle dashboard |

---

# Impact

## Quality Control (QC)

- Automated defect detection
- Objective visual inspection
- Reduced human error
- Traceable maintenance records
- Improved audit readiness

---

## Operations

- Faster inspection workflows
- Automated repair notifications
- Reduced turnaround time
- Improved fleet availability
- Technician productivity improvement

---

# Future Enhancements

- ROS2 Integration
- Drone Digital Twin
- Edge AI Deployment
- Multi-UAV Fleet Dashboard
- Cloud Analytics
- Automatic Maintenance Scheduling
- Predictive Failure Alerts
- Fleet Health Scoring
- Drone Hub Air-Traffic Management

---

# Author

**Himanshu Raj**

AI Engineer | Computer Vision Engineer | Robotics & Edge AI Engineer

Specializing in

- Artificial Intelligence
- Computer Vision
- UAV Systems
- Robotics
- Edge AI
- Android Development
- Predictive Maintenance
- Autonomous Inspection Systems

---

© 2025 All Rights Reserved.
# Author: Himanshu Raj
