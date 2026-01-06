
        // Init grids
        const pv = document.getElementById('pixel-view');
        const lv = document.getElementById('latent-view');
        for(let i=0; i<100; i++) { const d=document.createElement('div'); d.className='px'; pv.appendChild(d); }
        for(let i=0; i<16; i++) { const d=document.createElement('div'); d.className='px'; d.style.background='#f59e0b'; lv.appendChild(d); }
        
        function compress(v) {
            const scale = 1 - (v/100);
            pv.style.opacity = scale;
            lv.style.transform = `scale(${1 + (v/50)})`;
            lv.style.opacity = v/100;
            
            document.getElementById('comp-label').innerText = v > 50 ? "Latent Space (Compressed)" : "Pixel Space (Original)";
        }
        