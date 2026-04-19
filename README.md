# Multi-Modal Driver Safety System using Computer Vision

> Real-time perception and driver monitoring pipeline for proactive safety assessment

## 🎥 Demo Video

A short demonstration showcasing real-time road perception and driver monitoring:

👉 [Watch Demo Video](https://drive.google.com/file/d/1kuZ1te90d-X35TJJEIIgVpYfnKWb3DwA/view?usp=sharing)

---

## Overview

This project implements a real-time **Advanced Driver Assistance System (ADAS)** with integrated driver state monitoring, designed to detect and respond to unsafe driving conditions.

The system processes a forward-facing road video alongside a live driver camera feed, enabling synchronized multi-stream analysis of environment and driver behavior.

This implementation demonstrates how multi-modal perception can be combined to improve driver safety in real-time scenarios.

The system highlights how combining environmental perception with driver state analysis can enhance decision-making in intelligent driving systems.

---

## Key Capabilities

### Road Perception

* Real-time object detection using YOLOv8
* Lane detection with adaptive handling for day and low-light conditions
* Lane departure detection based on lane center estimation
* Approximate distance estimation using bounding box geometry

---

### Driver Monitoring

* Face and eye detection using OpenCV Haar cascades
* Drowsiness detection based on eye closure duration
* Continuous monitoring using webcam input

---

### Risk Assessment & Alerts

* Rule-based risk scoring combining multiple signals
* Independent triggers for:

  * High environmental risk
  * Driver drowsiness
* Alert mechanisms:

  * Audio feedback
  * Telegram notifications

---

### Output & Visualization

* Dual-stream visualization:

  * ADAS (road view)
  * Driver monitoring (webcam)
* Annotated output video saved as `output.mp4`

---

## Architecture

The system follows a modular pipeline:

* **Perception Layer**

  * Object detection (YOLOv8)
  * Lane detection (OpenCV)

* **Driver Monitoring**

  * Face and eye detection
  * Temporal analysis for drowsiness

* **Decision Layer**

  * Risk scoring
  * Alert triggering logic

---

## Technology Stack

* Python
* OpenCV
* Ultralytics YOLOv8
* NumPy
* Telegram Bot API

---

## Project Structure

```
project-root/
│
├── main.py
├── fatigue_detection.py (optional standalone module)
├── requirements.txt
├── README.md
```

---

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the system:

```bash
python main.py
```

---

## Input Requirements

* A road video named `road.mp4` must be placed in the project root directory
* The system processes this video for object and lane detection
* Any forward-facing driving footage can be used for testing

---

## Configuration

### Telegram Alerts

Update the following values in the code:

```python
TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
```

---

## Design Considerations

* Lightweight pipeline for near real-time execution
* Modular design enabling independent components
* Configurable thresholds for risk and drowsiness detection
* Simulates multi-input perception (environment + driver)

---

## Potential Enhancements

* Deep learning-based driver monitoring
* Multi-object tracking improvements
* Integration with mobile applications
* Cloud-based monitoring dashboard
* Edge deployment optimization

---

## Summary

This system demonstrates a practical implementation of a **multi-stream driver safety pipeline**, combining environmental perception and driver monitoring to support real-time decision-making in assisted driving scenarios.
