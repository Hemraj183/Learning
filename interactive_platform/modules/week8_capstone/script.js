
        function activate(type) {
            document.getElementById('path-llm').className = '';
            document.getElementById('path-diff').className = '';
            
            if(type === 'text') document.getElementById('path-llm').className = 'active-branch';
            else document.getElementById('path-diff').className = 'active-branch';
        }
        