document.addEventListener("DOMContentLoaded", () => {

  /* ===================== DISEASE PREDICTION ===================== */
  const predictForm = document.getElementById("predict-form");
  if (predictForm) {
    predictForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const fileInput = predictForm.querySelector('input[type="file"]');
      if (!fileInput || !fileInput.files.length) {
        alert("Please select an image");
        return;
      }

      const formData = new FormData();
      formData.append("file", fileInput.files[0]);

      const resultBox = document.getElementById("predict-result");
      resultBox.innerHTML = "⏳ Predicting...";

      try {
        const res = await fetch("/api/predict_disease", {
          method: "POST",
          body: formData
        });

        const data = await res.json();

        if (!res.ok) {
          resultBox.innerHTML = "❌ Error occurred";
          return;
        }

        const r = data.result;
        resultBox.innerHTML = `
          <b>Plant:</b> ${r.plant}<br>
          <b>Disease:</b> ${r.disease}<br>
          <b>Confidence:</b> ${r.confidence ?? "N/A"}
        `;
      } catch (err) {
        resultBox.innerHTML = "❌ Upload failed";
        console.error(err);
      }
    });
  }

  /* ===================== CROP RECOMMENDATION ===================== */
  const cropForm = document.getElementById("crop-form");
  if (cropForm) {
    cropForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const payload = {
        soil_ph: parseFloat(cropForm.soil_ph.value || 0),
        rainfall: parseFloat(cropForm.rainfall.value || 0),
        temperature: parseFloat(cropForm.temperature.value || 0)
      };

      const out = document.getElementById("crop-result");
      out.innerHTML = "⏳ Processing...";

      const res = await fetch("/api/crop_recommendation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      out.innerHTML = "🌾 Recommended Crop: <b>" + data.result.recommended_crop + "</b>";
    });
  }

  /* ===================== FERTILIZER ===================== */
  const fertForm = document.getElementById("fert-form");
  if (fertForm) {
    fertForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const payload = {
        soil_ph: parseFloat(fertForm.soil_ph.value || 0),
        nitrogen: parseFloat(fertForm.nitrogen.value || 0),
        phosphorus: parseFloat(fertForm.phosphorus.value || 0),
        potassium: parseFloat(fertForm.potassium.value || 0)
      };

      const out = document.getElementById("fert-result");
      out.innerHTML = "⏳ Calculating...";

      const res = await fetch("/api/fertilizer_advice", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      out.innerHTML = "🧪 Fertilizer: <b>" + data.result.recommendation + "</b>";
    });
  }

  /* ===================== AI ASSISTANT ===================== */
  const aiForm = document.getElementById("ai-form");
  if (aiForm) {
    aiForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const msg = aiForm.query.value;
      const out = document.getElementById("ai-result");
      out.innerHTML = "⏳ Thinking...";

      const res = await fetch("/api/assistant_chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: msg })
      });

      const data = await res.json();
      out.innerHTML = "🤖 " + data.response;
    });
  }

  /* ===================== SUBSIDIES ===================== */
  const subForm = document.getElementById("sub-form");
  if (subForm) {
    subForm.addEventListener("submit", async (e) => {
      e.preventDefault();

      const out = document.getElementById("sub-result");
      out.innerHTML = "⏳ Loading...";

      const res = await fetch("/api/subsidies", {
        method: "POST"
      });

      const data = await res.json();
      out.innerHTML = "<pre>" + JSON.stringify(data.schemes, null, 2) + "</pre>";
    });
  }

});
