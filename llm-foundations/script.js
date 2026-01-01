// ===================================
// Transformer Engine: Interactive Logic
// ===================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCheckpoints();
    initCopyButtons();
    initAttentionViz();
    initQKVViz();
    initPEViz();
    initPythonTerminal();
});

// ===================================
// Shared UI Utilities
// ===================================
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-links a');
    const sections = document.querySelectorAll('.section');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            const href = link.getAttribute('href');
            if (href.startsWith('#')) {
                e.preventDefault();
                const target = document.querySelector(href);
                if (target) {
                    target.scrollIntoView({ behavior: 'smooth' });
                }
            }
        });
    });

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                navLinks.forEach(link => {
                    link.classList.toggle('active', link.getAttribute('href') === `#${entry.target.id}`);
                });
            }
        });
    }, { threshold: 0.3 });

    sections.forEach(s => observer.observe(s));
}

function initCheckpoints() {
    const checkboxes = document.querySelectorAll('.checkpoint-input');
    const progressRing = document.querySelector('.progress-ring-circle');
    const progressText = document.querySelector('.progress-text');

    const updateProgress = () => {
        const total = checkboxes.length;
        const checked = [...checkboxes].filter(c => c.checked).length;
        const percentage = Math.round((checked / total) * 100);

        const circumference = 163;
        progressRing.style.strokeDashoffset = circumference - (percentage / 100) * circumference;
        progressText.textContent = `${percentage}%`;

        localStorage.setItem('transformer-progress', JSON.stringify([...checkboxes].map(c => c.checked)));
    };

    const saved = JSON.parse(localStorage.getItem('transformer-progress') || '[]');
    checkboxes.forEach((c, i) => {
        c.checked = saved[i] || false;
        c.addEventListener('change', updateProgress);
    });
    updateProgress();
}

function initCopyButtons() {
    document.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const code = document.getElementById(btn.dataset.copy).innerText;
            navigator.clipboard.writeText(code);
            btn.textContent = 'Copied!';
            setTimeout(() => btn.textContent = 'Copy', 2000);
        });
    });
}

// ===================================
// Interactive Visualizations
// ===================================

function initAttentionViz() {
    const tokens = document.querySelectorAll('.token');
    const matrix = document.getElementById('attention-matrix');
    const numTokens = 11;

    // Create matrix cells
    for (let i = 0; i < numTokens * numTokens; i++) {
        const cell = document.createElement('div');
        cell.className = 'matrix-cell';
        matrix.appendChild(cell);
    }

    const cells = document.querySelectorAll('.matrix-cell');

    tokens.forEach((token, idx) => {
        token.addEventListener('mouseover', () => {
            tokens.forEach(t => t.classList.remove('active'));
            token.classList.add('active');

            // Simulate attention weights (randomish but biased)
            cells.forEach((cell, cellIdx) => {
                const row = Math.floor(cellIdx / numTokens);
                const col = cellIdx % numTokens;

                let weight;
                if (row === idx) {
                    // This row represents what the selected token attends to
                    weight = Math.random() * 0.4;
                    if (col === idx) weight = 0.8; // Self-attention
                    if (idx === 7 && (col === 1 || col === 10)) weight = 0.9; // "it" attends to "animal"/"tired"
                } else {
                    weight = 0.05;
                }

                cell.style.background = `rgba(16, 185, 129, ${weight})`;
            });
        });
    });
}

function initQKVViz() {
    const container = document.getElementById('qkv-viz');
    container.innerHTML = `
        <div class="qkv-controls" style="display: flex; gap: 20px; align-items: center; margin-bottom: 20px;">
            <button class="viz-btn" id="qkv-step">Next Step: Query Match</button>
            <div id="qkv-desc" style="font-size: 0.9rem; color: var(--accent-emerald);">Click to simulate the QKV flow...</div>
        </div>
        <div class="qkv-animation" style="position: relative; width: 100%; height: 100px; background: rgba(0,0,0,0.3); border-radius: 8px; display: flex; justify-content: space-around; align-items: center;">
            <div id="q-node" style="padding: 10px; border: 2px solid var(--accent-blue); border-radius: 50%;">Q</div>
            <div id="k-node" style="padding: 10px; border: 2px solid var(--accent-emerald); border-radius: 50%;">K</div>
            <div id="score-node" style="padding: 10px; opacity: 0;">Softmax</div>
            <div id="v-node" style="padding: 10px; border: 2px solid var(--accent-purple); border-radius: 50%;">V</div>
        </div>
    `;

    let step = 0;
    const btn = document.getElementById('qkv-step');
    const desc = document.getElementById('qkv-desc');
    const nodes = {
        q: document.getElementById('q-node'),
        k: document.getElementById('k-node'),
        s: document.getElementById('score-node'),
        v: document.getElementById('v-node')
    };

    btn.addEventListener('click', () => {
        step = (step + 1) % 4;
        if (step === 1) {
            nodes.q.style.transform = "translateX(50px)";
            nodes.k.style.transform = "translateX(-50px)";
            desc.textContent = "Query matches against Key...";
        } else if (step === 2) {
            nodes.s.style.opacity = "1";
            desc.textContent = "Softmax computes attention scores.";
        } else if (step === 3) {
            nodes.v.style.transform = "scale(1.5)";
            desc.textContent = "Scores weight the Value (V). Result produced!";
        } else {
            Object.values(nodes).forEach(n => { n.style.transform = ""; n.style.opacity = step === 0 && n === nodes.s ? "0" : "1"; });
            desc.textContent = "Click to start over.";
        }
    });
}

function initPEViz() {
    const canvas = document.getElementById('pe-canvas');
    const ctx = canvas.getContext('2d');

    let time = 0;
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 2;
        ctx.beginPath();

        for (let x = 0; x < canvas.width; x++) {
            const y = canvas.height / 2 + Math.sin(x * 0.05 + time) * 30 + Math.cos(x * 0.02 + time * 0.5) * 20;
            if (x === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        time += 0.05;
        requestAnimationFrame(animate);
    }
    animate();
}

// ===================================
// Python Terminal logic (Ported)
// ===================================

let pyodide = null;

async function initPythonTerminal() {
    const terminal = document.getElementById('python-terminal');
    const toggleBtn = document.getElementById('terminal-toggle');
    const runBtn = document.getElementById('run-code');
    const status = document.getElementById('terminal-status');
    const output = document.getElementById('terminal-output');
    const codeInput = document.getElementById('python-code');
    const snippets = document.getElementById('example-snippets');

    toggleBtn.addEventListener('click', () => {
        terminal.classList.toggle('active');
        if (!pyodide) loadPy();
    });

    document.getElementById('close-terminal').addEventListener('click', () => terminal.classList.remove('active'));
    document.getElementById('minimize-terminal').addEventListener('click', () => terminal.style.height = terminal.offsetHeight < 100 ? 'auto' : '50px');

    async function loadPy() {
        status.textContent = "Loading Python...";
        try {
            pyodide = await loadPyodide();
            await pyodide.loadPackage("numpy");
            status.textContent = "Ready";
            status.style.color = "#10b981";
        } catch (e) {
            status.textContent = "Error";
            status.style.color = "#ef4444";
        }
    }

    runBtn.addEventListener('click', async () => {
        if (!pyodide) return;
        const code = codeInput.value;
        try {
            output.innerHTML = "";
            pyodide.setStdout({ batched: (str) => output.innerHTML += `<div>${str}</div>` });
            await pyodide.runPythonAsync(code);
        } catch (e) {
            output.innerHTML += `<div style="color:#ef4444">${e.message}</div>`;
        }
    });

    snippets.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'attention') {
            codeInput.value = `import numpy as np\n\ndef softmax(x):\n    e_x = np.exp(x - np.max(x))\n    return e_x / e_x.sum()\n\n# Simulate Q, K, V\nQ = np.array([1, 0, 1])\nK = np.array([1, 1, 0])\nV = np.array([10, 20, 30])\n\nscore = np.dot(Q, K)\nweight = softmax([score, 0, 0])\nprint(f"Attention Weight: {weight[0]:.2f}")\nprint(f"Result: {weight[0]*V[0]:.2f}")`;
        } else if (val === 'transformer') {
            codeInput.value = `import numpy as np\n\nclass TransformerBlock:\n    def __init__(self):\n        pass\n\n    def forward(self, x):\n        # 1. Attention\n        attn = x + 0.1 \n        # 2. Add & Norm\n        x = attn + x\n        # 3. Feed Forward\n        ff = x * 2\n        # 4. Add & Norm\n        return ff + x\n\nblock = TransformerBlock()\nx = np.array([1.0, 2.0])\nprint(f"Input: {x}")\nprint(f"Output: {block.forward(x)}")`;
        }
    });
}
