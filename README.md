# CyPath 🚴‍♂️💻
**Dynamic Algorithmic Endurance Training Engine**

*A Final-Year Computer Science Project — Bournemouth University*
* **Developer:** Gustavo Miranda
* **Supervisor:** Benjamin Gorman

---

## 📖 About the Project
Amateur cyclists struggle to safely train for long-distance endurance events (like a 100km ride). Traditional, static PDF training plans are flawed: the moment a user misses a single workout due to real-world interruptions, the entire schedule breaks, leading to dangerous overtraining or ineffective undertraining. 

**CyPath** solves this by acting as a dynamic, algorithm-driven scheduling engine. It tracks hidden physical fatigue using sports-science mathematics and automatically recalculates the user's future path if a session is skipped, ensuring they still peak on race day safely.

## ✨ Key Features
* **The Banister Model (Readiness):** Calculates daily physiological readiness using the formula `Readiness = Fitness - Fatigue`.
* **Dynamic Re-routing:** Acts as a Constraint-Satisfaction Problem (CSP) solver. If a session is missed, the system intelligently redistributes the missed workload across the remaining weeks rather than forcing a dangerous "catch-up" day.
* **Mobile-First UI:** A clean, responsive dashboard designed to mimic a native mobile application environment.

## 🛠️ Tech Stack
* **Backend:** Python (Core algorithm and mathematical modeling)
* **Framework:** Flask / FastAPI (API and routing)
* **Frontend:** HTML5, CSS3, Vanilla JavaScript

## 🚀 How to Run Locally

To test the application on your local machine, follow these steps:

**1. Clone the repository:**
```bash
git clone [https://github.com/GustavoMiranda273/CyPath.git](https://github.com/GustavoMiranda273/CyPath.git)
cd CyPath