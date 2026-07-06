---
source: projects/fifa-graphs
topic: projects
last_updated: 2026-07-05
---

# FIFA World Cup Graphs Site

## What it is

An interactive data visualization platform for FIFA World Cup statistics,
built with React on the front end and Python on the back end.

## How it works

The React UI has dynamic filtering and real-time graph updates. The backend
pipeline uses pandas and Matplotlib to query the data and generate plots,
which cut manual processing time by about 95%. I originally explored MySQL for
the match statistics, then simplified the project by moving to CSV files.
