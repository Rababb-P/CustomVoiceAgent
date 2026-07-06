---
source: projects/smart-bin
topic: projects
last_updated: 2026-07-05
---

# Smart Bin

## What it is

Smart Bin is an automated waste-sorting bin I built at StarterHacks with
Python, TensorFlow, and Arduino hardware.

## How it works

I took the pictures, labeled them, configured the training pipeline, and
fine-tuned a TensorFlow model to detect recyclables versus garbage at 85%
accuracy. The model talks to Arduino-controlled servo motors over serial
communication, so the bin physically sorts waste from recycling in real time.
