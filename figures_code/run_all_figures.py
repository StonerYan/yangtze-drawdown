"""Run all 9 figure scripts in sequence to regenerate PNG and PDF figures."""
import os, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = [
    "fig01_concept.py",
    "fig02_study_area.py",
    "fig03_contrast.py",
    "fig04_sensitivity.py",
    "fig05_longrecord.py",
    "fig06_coupling.py",
    "fig07_remaining.py",
    "fig08_exposure.py",
    "fig_ga.py",
]

for s in SCRIPTS:
    print("Running", s, "...")
    r = subprocess.run([sys.executable, "-X", "utf8", os.path.join(HERE, s)], check=True)
print("All figures generated successfully.")
