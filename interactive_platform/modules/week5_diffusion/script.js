
        const ctx = document.getElementById('c').getContext('2d');
        function draw(v) {
            ctx.fillStyle='white'; ctx.fillRect(0,0,200,200);
            ctx.fillStyle='black'; ctx.font='80px Arial'; ctx.fillText('AI', 60, 130);
            
            const img = ctx.getImageData(0,0,200,200);
            const d = img.data;
            for(let i=0; i<d.length; i+=4) {
                if(Math.random()*100 < v) {
                    const g = Math.random()*255;
                    d[i]=g; d[i+1]=g; d[i+2]=g;
                }
            }
            ctx.putImageData(img,0,0);
        }
        draw(0);
        