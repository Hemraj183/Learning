
        function highlight(level) {
            document.getElementById(`dec-${level}`).style.fill = '#fffff';
            document.getElementById(`dec-${level}`).style.stroke = '#fff';
            document.getElementById(`dec-${level}`).style.strokeWidth = '3px';
            document.getElementById(`skip-${level}`).style.opacity = '1';
            document.getElementById(`skip-${level}`).style.stroke = '#f43f5e';
        }
        function reset() {
            for(let i=1; i<=3; i++) {
                document.getElementById(`dec-${i}`).style.fill = '#f59e0b';
                document.getElementById(`dec-${i}`).style.stroke = 'none';
                document.getElementById(`skip-${i}`).style.opacity = '0.3';
                document.getElementById(`skip-${i}`).style.stroke = '#fff';
            }
        }
        