# Solo Parent Decision Support System (DSS)

An AI-powered web application to assess eligibility for Solo Parent benefits under **Republic Act 11861** (Solo Parent Welfare Act).

## Features

✨ **Comprehensive Assessment Form**
- Section I: Identifying Information (personal details, income, occupation)
- Section II: Family Composition (dynamic table for household members)
- Section III: Solo Parent Status (reason and circumstances)

✨ **Intelligent Eligibility Assessment**
- Rule-based logic aligned with RA 11861 criteria
- Three-tier eligibility status: Eligible, Needs Verification, Not Eligible
- Income threshold calculations based on family size

✨ **PDF Report Generation**
- Professional questionnaire-formatted PDF
- Includes applicant info, family composition, and results
- Personalized recommendations based on eligibility status
- Auto-download functionality

✨ **Professional UI**
- Dark navy color scheme with dark red accents
- Responsive Bootstrap design
- Font Awesome icons
- Mobile-friendly interface

## Tech Stack

- **Backend:** Flask (Python)
- **Frontend:** Bootstrap 5, Jinja2 Templates
- **PDF Generation:** ReportLab
- **Data Processing:** Pandas, NumPy

## Installation

### Prerequisites
- Python 3.8+
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
├── app.py                    # Main Flask application
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
│
├── templates/
│   ├── base.html            # Base template with navbar/footer
│   ├── index.html           # Homepage with benefits showcase
│   ├── assessment_form.html # Application form (3 sections)
│   └── result.html          # Results page with PDF download
│
├── static/
│   ├── css/
│   │   └── style.css        # Custom styling
│   └── js/
│       └── script.js        # (Optional) JavaScript utilities
│
├── data/
│   └── synthetic_solo_parent_dataset.csv  # Sample data
│
├── model/
│   ├── solo_parent_model.pkl    # ML model (future)
│   └── encoders.pkl             # Label encoders (future)
│
└── reports/
    └── (Generated PDF reports saved here)
```

## Usage

### 1. Fill Assessment Form
- Enter personal information
- Add family members (name, relationship, age, education)
- Provide monthly income details
- Select solo parent status/reason

### 2. Submit & Get Results
- System automatically assesses eligibility
- Three possible outcomes:
  - ✅ **Eligible** - Qualifies for all RA 11861 benefits
  - ⚠️ **Needs Verification** - Requires additional documents
  - ❌ **Not Eligible** - Doesn't meet current criteria

### 3. Download Report
- Click "Download PDF Report" button
- Save professionally formatted questionnaire
- Includes eligibility result and recommendations

## RA 11861 Benefits (When Eligible)

- 💰 **Monthly Subsidy:** ₱1,500 per child
- 🛍️ **VAT Exemption:** Tax benefits on goods/services
- 📚 **Educational Support:** Scholarships and school assistance
- 🏛️ **Priority Services:** Access to government assistance

## Assessment Criteria

**Eligibility Requirements:**
- Valid solo parent status (widowed, abandoned, separated, or single parent)
- At least 1 dependent child
- Family income below threshold (~₱30,000 + ₱5,000 per child)
- Age 18 or older

## Future Enhancements

- 🤖 Machine learning model for predictive eligibility
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

## Support

For issues or questions, please create an issue in the GitHub repository or contact the development team.

## Disclaimer

This is an automated assessment tool based on RA 11861 criteria. Final determination of eligibility is made by the Department of Social Welfare and Development (DSWD) upon submission of complete documents and verification interview.

---

**Version:** 1.0.0  
**Last Updated:** May 2026  
**Status:** Active Development
