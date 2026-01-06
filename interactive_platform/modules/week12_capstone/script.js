
        function ping() {
            const s = document.getElementById('sys-stat');
            s.style.transform = 'scale(1.5)';
            s.style.boxShadow = '0 0 50px white';
            setTimeout(() => {
                s.style.transform = 'scale(1)';
                s.style.boxShadow = '0 0 20px #8b5cf6';
            }, 300);
        }
        