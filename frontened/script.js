const API_URL = "http://localhost:8000/predict";

function predict() {
    const input = document.getElementById("imageInput");
    const file = input.files[0];

    if (!file) {
        alert("Please upload an image");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    document.getElementById("resultCard").style.display = "block";
    document.getElementById("predictionText").innerText = "Predicting...";
    document.getElementById("confidenceText").innerText = "";

    fetch(API_URL, {
        method: "POST",
        body: formData
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("previewImage").src = URL.createObjectURL(file);
        document.getElementById("predictionText").innerText =
            `Prediction: ${data.class}`;
        document.getElementById("confidenceText").innerText =
            `Confidence: ${data.confidence}%`;
    })
    .catch(() => {
        alert("Prediction failed");
    });
}
