document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData);

    try {
        const response = await fetch('https://localhost:5500/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });
        const result = await response.json();
        document.getElementById('result').classList.remove('hidden');
        document.getElementById('prediction').textContent = 
            `$${result.prediction.toLocaleString()}`;
    } catch (error) {
        console.error('Error:', error);
    }
});