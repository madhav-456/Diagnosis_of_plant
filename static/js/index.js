// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
  // detect page by id presence

  // Disease prediction form
  const predictForm = document.getElementById('predict-form');
  if (predictForm) {
    predictForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = predictForm.querySelector('input[name="file"]');
      if (!fileInput.files.length) return alert('Select an image');

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);

      const btn = predictForm.querySelector('button[type="submit"]');
      const prevText = btn.textContent;
      btn.disabled = true; btn.textContent = 'Predicting...';

      try {
        const res = await fetch('/api/predict_disease', { method: 'POST', body: formData });
        const data = await res.json();
        const out = document.getElementById('predict-result');
        if (res.ok) {
          out.innerHTML = `<pre>${JSON.stringify(data.result, null, 2)}</pre>`;
        } else {
          out.innerHTML = `<div class="result">Error: ${data.error || JSON.stringify(data)}</div>`;
        }
      } catch (err) {
        alert('Upload failed: ' + err);
      } finally {
        btn.disabled = false; btn.textContent = prevText;
      }
    });
  }

  // Fertilizer form
  const fertForm = document.getElementById('fert-form');
  if (fertForm) {
    fertForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const payload = {
        crop: fertForm.crop.value,
        soil_ph: parseFloat(fertForm.soil_ph.value || 0),
        nitrogen: parseFloat(fertForm.nitrogen.value || 0),
        phosphorus: parseFloat(fertForm.phosphorus.value || 0),
        potassium: parseFloat(fertForm.potassium.value || 0),
        symptoms: fertForm.symptoms.value || ''
      };
      const res = await fetch('/api/fertilizer_advice', {
        method:'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      document.getElementById('fert-result').innerHTML = `<pre>${JSON.stringify(data.result, null, 2)}</pre>`;
    });
  }

  // Crop recommendation
  const cropForm = document.getElementById('crop-form');
  if (cropForm) {
    cropForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const payload = {
        soil_ph: parseFloat(cropForm.soil_ph.value || 0),
        rainfall: parseFloat(cropForm.rainfall.value || 0),
        temperature: parseFloat(cropForm.temperature.value || 0),
        location: cropForm.location.value || ''
      };
      const res = await fetch('/api/crop_recommendation', {
        method:'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      document.getElementById('crop-result').innerHTML = `<pre>${JSON.stringify(data.result, null, 2)}</pre>`;
    });
  }

  // AI assistant
  const aiForm = document.getElementById('ai-form');
  if (aiForm) {
    aiForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const payload = { query: aiForm.query.value };
      const res = await fetch('/api/ai_assistant', {
        method:'POST',
        headers: {'Content-Type':'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      document.getElementById('ai-result').innerHTML = `<pre>${JSON.stringify(data.result, null, 2)}</pre>`;
    });
  }

  // Subsidy lookup
  const subForm = document.getElementById('sub-form');
  if (subForm) {
    subForm.addEventListener('submit', async (e) => {
      e.preventDefault();
      const country = subForm.country.value || 'IN';
      const crop = subForm.crop.value || '';
      const res = await fetch(`/api/subsidies?country=${encodeURIComponent(country)}&crop=${encodeURIComponent(crop)}`);
      const data = await res.json();
      document.getElementById('sub-result').innerHTML = `<pre>${JSON.stringify(data.result, null, 2)}</pre>`;
    });
  }
});
