document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('cardio-form');
    const btnAssess = document.getElementById('btn-assess');
    const spinner = document.getElementById('form-spinner');
    const btnText = btnAssess.querySelector('.btn-text');

    // UI Result Elements
    const riskVal = document.getElementById('risk-val');
    const riskTier = document.getElementById('risk-tier');
    const riskCircle = document.querySelector('.risk-circle');
    const badgeRec = document.getElementById('badge-recommendation');
    const metricCategory = document.getElementById('metric-category');
    const meterRiskFill = document.getElementById('meter-risk-fill');
    const metricTier = document.getElementById('metric-tier');
    const guidanceText = document.getElementById('guidance-text');
    const factorList = document.getElementById('factor-list');

    // Clinical Presets
    const presets = {
        low: {
            age: 36, sex: 0, trestbps: 112, chol: 175, thalach: 172,
            oldpeak: 0.1, cp: 1, fbs: 0, exang: 0, restecg: 0, slope: 2, ca: 0, thal: 1
        },
        moderate: {
            age: 52, sex: 1, trestbps: 135, chol: 228, thalach: 142,
            oldpeak: 1.4, cp: 0, fbs: 0, exang: 0, restecg: 1, slope: 1, ca: 1, thal: 2
        },
        high: {
            age: 65, sex: 1, trestbps: 165, chol: 295, thalach: 115,
            oldpeak: 3.4, cp: 0, fbs: 1, exang: 1, restecg: 1, slope: 1, ca: 2, thal: 3
        }
    };

    function applyPreset(data) {
        for (const [key, value] of Object.entries(data)) {
            const el = document.getElementById(key);
            if (el) el.value = value;
        }
        triggerAssessment();
    }

    document.getElementById('preset-low').addEventListener('click', () => applyPreset(presets.low));
    document.getElementById('preset-moderate').addEventListener('click', () => applyPreset(presets.moderate));
    document.getElementById('preset-high').addEventListener('click', () => applyPreset(presets.high));

    async function triggerAssessment() {
        btnAssess.disabled = true;
        spinner.style.display = 'inline-block';
        btnText.textContent = 'Analyzing Clinical Biomarkers...';

        const patientData = {
            age: parseInt(document.getElementById('age').value),
            sex: parseInt(document.getElementById('sex').value),
            trestbps: parseFloat(document.getElementById('trestbps').value),
            chol: parseFloat(document.getElementById('chol').value),
            thalach: parseFloat(document.getElementById('thalach').value),
            oldpeak: parseFloat(document.getElementById('oldpeak').value),
            cp: parseInt(document.getElementById('cp').value),
            fbs: parseInt(document.getElementById('fbs').value),
            exang: parseInt(document.getElementById('exang').value),
            restecg: parseInt(document.getElementById('restecg').value),
            slope: parseInt(document.getElementById('slope').value),
            ca: parseInt(document.getElementById('ca').value),
            thal: parseInt(document.getElementById('thal').value)
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(patientData)
            });

            const res = await response.json();
            if (res.status === 'success') {
                const data = res.data;

                // Animate risk value
                const currentVal = parseFloat(riskVal.textContent) || 0;
                animateFloatValue(riskVal, currentVal, data.estimated_10yr_risk_percent, 600);

                riskTier.textContent = data.risk_category;
                riskTier.style.color = data.color_code;
                riskCircle.style.borderColor = data.color_code;
                riskCircle.style.boxShadow = `0 0 25px ${data.color_code}40`;

                // Badge Recommendation
                badgeRec.textContent = data.clinical_recommendation.replace('_', ' ');
                badgeRec.className = `decision-badge ${data.clinical_recommendation.toLowerCase().replace('_', '-')}`;

                // Metrics
                metricCategory.textContent = data.risk_category;
                metricCategory.style.color = data.color_code;
                meterRiskFill.style.width = `${Math.min(data.estimated_10yr_risk_percent, 100)}%`;
                meterRiskFill.style.background = data.color_code;

                metricTier.textContent = data.aha_tier;

                // Guidance
                guidanceText.textContent = data.actionable_guidance;

                // Clinical Risk Biomarkers
                factorList.innerHTML = '';
                if (data.clinical_drivers && data.clinical_drivers.length > 0) {
                    data.clinical_drivers.forEach(driver => {
                        const li = document.createElement('li');
                        li.textContent = driver;
                        factorList.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = 'No acute adverse clinical cardiovascular biomarkers detected.';
                    factorList.appendChild(li);
                }
            }
        } catch (err) {
            console.error('Prognosis API error:', err);
            alert('Failed to connect to cardiovascular prognosis service.');
        } finally {
            btnAssess.disabled = false;
            spinner.style.display = 'none';
            btnText.textContent = 'Compute 10-Year ASCVD Risk';
        }
    }

    function animateFloatValue(obj, start, end, duration) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const current = (progress * (end - start) + start).toFixed(1);
            obj.innerHTML = `${current}%`;
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = `${end.toFixed(1)}%`;
            }
        };
        window.requestAnimationFrame(step);
    }

    form.addEventListener('submit', (e) => {
        e.preventDefault();
        triggerAssessment();
    });

    // Run initial on page load
    triggerAssessment();
});
