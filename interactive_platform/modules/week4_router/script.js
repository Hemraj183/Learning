
        function route() {
            const t = document.getElementById('q').value.toLowerCase();
            document.querySelectorAll('.exp').forEach(e=>e.classList.remove('active-e'));
            
            let id = 'e1';
            if (t.includes('code') || t.includes('function') || t.includes('print')) id='e2';
            else if (t.match(/[0-9]/)) id='e3';
            
            document.getElementById(id).classList.add('active-e');
        }
        