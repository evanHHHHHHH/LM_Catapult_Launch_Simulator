# LM_Catapult_Launch_Simulator
Interactive tool for simulating projectile motion from a catapult launch, including aerodynamics (Cd/Cl by AOA), thrust polynomial, drag/lift forces, and RK4 numerical integration.

## Features
- Adjustable params: v0, angle, height, mass, area, density.
- AOA cases from aero table.
- Validation vs. analytic solution.
- Trajectory plots & CSV export.
- Matches Omni Calculator for no-aero cases.

## Local Run
pip install -r requirements.txt
streamlit run app.py

## Deployed
[Live App](https://your-app-link.streamlit.app) – Sliders for real-time sims!
