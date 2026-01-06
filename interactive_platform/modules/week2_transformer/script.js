
        function visualizeAttention() {
            const svg = document.getElementById('attn-viz');
            svg.innerHTML = ''; 
            const words = ["The", "quick", "brown", "fox", "jumps", "over", "dog"];
            const padding = 70;
            
            words.forEach((w, i) => {
                const t1 = document.createElementNS("http://www.w3.org/2000/svg", "text");
                t1.setAttribute("x", i * padding + 40); t1.setAttribute("y", 50); t1.setAttribute("fill", "white"); t1.textContent = w; svg.appendChild(t1);
                
                const t2 = document.createElementNS("http://www.w3.org/2000/svg", "text");
                t2.setAttribute("x", i * padding + 40); t2.setAttribute("y", 250); t2.setAttribute("fill", "white"); t2.textContent = w; svg.appendChild(t2);
            });
            
            for(let i=0; i<words.length; i++) {
                for(let j=0; j<words.length; j++) {
                    const weight = Math.random();
                    if (weight > 0.6) {
                        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                        line.setAttribute("x1", i * padding + 50); line.setAttribute("y1", 60);
                        line.setAttribute("x2", j * padding + 50); line.setAttribute("y2", 230);
                        line.setAttribute("stroke", "#d946ef"); line.setAttribute("stroke-width", Math.pow(weight, 3) * 5); line.setAttribute("opacity", weight * 0.6);
                        svg.appendChild(line);
                    }
                }
            }
        }
        visualizeAttention(); // Init
        