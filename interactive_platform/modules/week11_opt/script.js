
        let q = false;
        function toggleQ() {
            q = !q;
            const b = document.getElementById('q-box');
            if(q) {
                b.style.filter = 'contrast(500%) grayscale(100%)'; // Mock quantization artifacts
                document.getElementById('q-btn').innerText = "Revert to FP32";
            } else {
                b.style.filter = 'none';
                document.getElementById('q-btn').innerText = "Apply INT8 Quantization";
            }
        }
        