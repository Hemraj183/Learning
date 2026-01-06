
        const c = document.getElementById('moe-canvas');
        const ctx = c.getContext('2d');
        let tokens = [];
        
        // Experts
        const experts = [
            {x: 500, y: 50, color: 'red'},
            {x: 500, y: 150, color: 'blue'}
        ];
        
        function spawnToken() {
            const isRed = Math.random() > 0.5;
            tokens.push({
                x: 0, y: 100, 
                target: isRed ? 0 : 1,
                color: isRed ? 'red' : 'blue',
                progress: 0
            });
        }
        
        function animate() {
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            ctx.fillRect(0,0,600,200);
            
            // Draw Experts
            experts.forEach(e => {
                ctx.fillStyle = e.color;
                ctx.fillRect(e.x, e.y, 40, 40);
            });
            
            // Update tokens
            tokens.forEach((t, i) => {
                t.progress += 0.01;
                const ex = experts[t.target];
                
                // Lerp
                const cx = (1-t.progress)*0 + t.progress*ex.x;
                const cy = (1-t.progress)*100 + t.progress*(ex.y+20);
                
                ctx.beginPath();
                ctx.arc(cx, cy, 5, 0, Math.PI*2);
                ctx.fillStyle = t.color;
                ctx.fill();
                
                if(t.progress >= 1) tokens.splice(i, 1);
            });
            
            requestAnimationFrame(animate);
        }
        animate();
        