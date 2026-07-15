# Solo Parent Decision Support System (DSS)

A web application to assess eligibility for Solo Parent benefits under **Republic Act 11861** (Expanded Solo Parents Welfare Act) and its Revised IRR.

## Features

✨ **Comprehensive Assessment Form**
- Section I: Identifying Information (personal details, income, occupation)
- Section II: Family Composition (dynamic table for household members)
- Section III: Solo Parent Status (reason and circumstances)
- Section IV: Benefit Screening (fields that determine which specific benefits apply)

✨ **Rules-Engine-Driven Eligibility Assessment**
- Eligibility, benefits, priority level, and rejection reason are all decided by `engine.py`, which reads every threshold, category, and benefit condition from `rules.json` — nothing is hardcoded in `app.py`
- A secondary ML model (Random Forest) scores each case independently for a confidence percentage and flags disagreement with the rules engine for manual review; it does not decide eligibility
- Independent income-outlier check flags applicants far outside the training dataset's income range for manual review

✨ **PDF Report Generation**
- Government-form-styled PDF (ReportLab)
- Includes applicant info, family composition, eligibility result, and the exact benefits granted
- Auto-download functionality

✨ **Professional UI**
- Dark navy color scheme with dark red accents
- Responsive Bootstrap design
- Font Awesome icons

## Tech Stack

- **Backend:** Flask (Python)
- **Rules engine:** Plain Python (`engine.py`), config-driven by `rules.json`
- **Frontend:** Bootstrap 5, Jinja2 Templates
- **PDF Generation:** ReportLab
- **ML (secondary signal only):** scikit-learn (Random Forest, calibrated)

## Installation

### Prerequisites
- Python 3.10+
- pip (Python package manager)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/[member-username]/solo-parent-dss.git
   cd solo-parent-dss
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **Mac/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

6. **Access the app**
   Open your browser and go to: **http://localhost:5000**

## Project Structure

```
solo-parent-dss/
├── app.py                    # Flask app: routes, form parsing, PDF generation
├── engine.py                 # Pure evaluate(applicant) rules engine — reads only rules.json
├── rules.json                # Single source of truth: categories, thresholds, benefit
│                              # conditions, priority scoring. Edit this, not the code,
│                              # when the law/thresholds change.
├── eval_accuracy.py          # Validates engine.py against data/solo_parent_dataset.csv,
│                              # reporting per-field accuracy (not one blended number)
├── train_model_fixed.py      # Trains/tunes the secondary ML confidence model
├── requirements.txt
├── .gitignore
│
├── templates/
│   ├── base.html                        # Base template with navbar/footer
│   ├── index.html                       # Homepage
│   ├── assessment_form.html             # Application form (4 sections)
│   ├── result.html                      # Results page with PDF download
│   └── _result_technical_details.html   # ML confidence / conflict details (collapsible)
│
├── static/
│   └── css/style.css        # Custom styling
│
├── data/
│   └── solo_parent_dataset.csv    # Labeled dataset used by both eval_accuracy.py
│                                    # and train_model_fixed.py
│
└── model/
    ├── solo_parent_model.pkl    # Trained ML model artifact (secondary confidence signal)
    ├── encoders.pkl
    └── model_metadata.pkl       # Training metrics and benchmark summary
```

## Usage

### 1. Fill Assessment Form
- Enter personal information
- Add family members (name, relationship, age, education, income)
- Provide monthly income details
- Select solo parent status/reason
- Answer the benefit-screening questions (Section IV)

### 2. Submit & Get Results
- The rules engine (`engine.evaluate()`) decides eligibility, applicable benefits, priority level, and — if not eligible — the specific reason
- Two possible outcomes:
  - ✅ **Eligible** — lists exactly which of the 6 RA 11861 benefits apply, plus a priority level (High/Medium/Low)
  - ❌ **Not Eligible** — shows the specific reason (invalid category, duration below the legal minimum, no qualifying dependent, etc.)
- A case may additionally be flagged **"Needs Verification"** if the secondary ML model disagrees with the rules engine, or if income is far outside the range the ML model was trained on — this is a manual-review flag, not a third eligibility outcome

### 3. Download Report
- Click "Download PDF Report" button
- Save professionally formatted, government-form-styled report

## RA 11861 Benefits (When Eligible)

Each benefit below is independently gated by its own condition in `rules.json` — being eligible as a solo parent does not automatically grant all six:

- 💰 **Cash Subsidy (Sec. 15a):** ₱1,000/month per solo parent (not per child), for those earning minimum wage or below, subject to fund availability
- 🛍️ **VAT Discount/Exemption (Sec. 15b):** 10% discount and VAT exemption on baby's milk, food, micronutrient supplements, sanitary diapers, and prescribed medicines/vaccines, for children aged 0–6, for solo parents earning under ₱250,000/year
- 🏥 **PhilHealth/NHIP Automatic Coverage (Sec. 15c):** for solo parents not already a formal PhilHealth member
- 💼 **Livelihood/Employment Priority (Sec. 15d):** for solo parents who are unemployed or part-time
- 🏛️ **Housing Priority (Sec. 15e):** for solo parents below the poverty line
- 📚 **Educational Support (Sec. 9):** for solo parents with a dependent currently enrolled in school

## Assessment Criteria

**Eligibility Requirements** (all decided by `rules.json`/`engine.py`, not hardcoded in `app.py`):
- A recognized solo-parent circumstance under RA 11861 (death of spouse, abandonment, legal/de facto separation, unmarried parent, spouse detained/incapacitated, annulment/nullity/divorce, guardian/adoptive/foster parent, relative caregiver, pregnant woman, OFW-related guardian)
- For time-gated circumstances, the reported duration must meet the Revised IRR's minimum: abandonment/separation/relative-caregiver ≥ 6 months, spouse detention ≥ 3 months, OFW-related ≥ 12 months
- At least 1 dependent
- Applicant is 18 or older

There is no blanket income cutoff for eligibility itself — income only gates which *specific benefits* (cash subsidy, VAT discount, housing priority) apply, per the thresholds above.

## Algorithm Assessment

`train_model_fixed.py` compares Decision Tree, Random Forest, and Logistic Regression (the three algorithms actually used in this project) on `data/solo_parent_dataset.csv`, tunes each via grid search, and selects the best performer that also meaningfully uses `Monthly_Income` in its decision (a safety check against a model that ignores income entirely). Run it to see current metrics:

```bash
python train_model_fixed.py
```

Note: this ML model is a **secondary confidence signal only** — actual eligibility decisions come from `engine.py`/`rules.json`, not the model. See `assess_eligibility_ml()` and `assess_eligibility()` in `app.py`.

## Future Enhancements

- 📊 Admin dashboard with analytics
- 🔐 Secure user accounts and application history
- 📱 Mobile app version
- 🌐 Multi-language support (Tagalog, English)
- 📧 Email notifications and reminders
- 🗂️ Document upload and management

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open-source and available under the MIT License.

## Disclaimer

This is an automated assessment tool based on RA 11861 criteria. Final determination of eligibility is made by the Department of Social Welfare and Development (DSWD) upon submission of complete documents and verification interview.

---

**Status:** Active Development
