
        function tokenize() {
            const text = document.getElementById('t-in').value;
            const container = document.getElementById('t-out');
            container.innerHTML = '';
            // Mock tokenizer logic
            const words = text.split(/([\s\S])/);
            let buf = "";
            for(let c of text) {
                buf+=c;
                if(Math.random()>0.5 || c===' ') {
                     const d = document.createElement('div');
                     d.className = 'token';
                     d.textContent = buf;
                     d.style.backgroundColor = `hsl(${Math.random()*360}, 60%, 20%)`;
                     container.appendChild(d);
                     buf="";
                }
            }
            if(buf) {
                const d = document.createElement('div'); d.className = 'token'; d.textContent = buf; d.style.backgroundColor = `hsl(${Math.random()*360}, 60%, 20%)`; container.appendChild(d);
            }
        }
        