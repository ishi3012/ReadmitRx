# 🩺 ReadmitRx

A production-grade ML system to **predict 30-day Emergency Department (ED) readmissions** and intelligently **route Community Health Worker (CHW) interventions**. Built for HealthTech deployment, ReadmitRx combines classification, clustering, and routing logic to triage high-risk patients and optimize outreach.

---

## 🚀 Project Goals

- Predict likelihood of 30-day ED readmission using ML classifiers
- Cluster patients by risk profiles to personalize CHW strategies
- Route limited CHW capacity using explainable, policy-driven logic
- Provide a clean, extensible platform for public health deployment

---

## 🧱 Tech Stack

- **Python 3.10**
- `scikit-learn`, `xgboost`, `lightgbm`, `pandas`, `matplotlib`
- `mypy`, `black`, `ruff`, `pytest`, `GitHub Actions`
- (Planned) `Streamlit` or `FastAPI` for CHW triage dashboard

---

## 📁 Repository Structure

```mermaid
readmitrx/ # ML logic: pipeline, clustering, scoring, routing
app/ # Frontend (Streamlit/FastAPI - WIP)
models/ # Saved model artifacts or schema definitions
scripts/ # Batch jobs, model training scripts
tests/ # Unit tests (pytest)
docs/ # Model cards, setup guides, README assets
.github/ # CI workflows

```

--

## 🛠️ Developer Setup

```bash
# Clone and install
git clone https://github.com/<your-username>/ReadmitRx.git
cd ReadmitRx
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run lint, typecheck, tests
make check
```
# 📄 License
MIT — see LICENSE file
