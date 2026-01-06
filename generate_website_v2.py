import os

BASE_DIR = "interactive_platform/modules"

# ==========================================
# 1. HTML TEMPLATE (Premium + Global Nav)
# ==========================================
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Deep Learning Mastery</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <!-- Global Progress Tracker (Hidden Metadata) -->
    <div id="course-metadata" data-current-week="{week_id}" style="display:none;"></div>

    <!-- Sidebar -->
    <nav class="sidebar">
        <div class="sidebar-header">
            <h2>{short_title}</h2>
            <div class="progress-container">
                 <div class="progress-bar"><div class="progress-fill" id="page-progress"></div></div>
                 <span class="progress-text" id="progress-text">0% Complete</span>
            </div>
            <a href="../../index.html" class="back-link">← Course Dashboard</a>
        </div>
        
        <div class="nav-scroll-area">
            <h3 class="nav-section-title">This Module</h3>
            <ul class="nav-links local-nav">
                <li><a href="#intro">📖 Introduction</a></li>
                <li><a href="#theory">🧠 Theory & Math</a></li>
                <li><a href="#interactive">🔬 Interactive Lab</a></li>
                <li><a href="#code">💻 Implementation</a></li>
                <li><a href="#project">🎯 Project</a></li>
            </ul>

            <h3 class="nav-section-title" style="margin-top:20px;">Course Map</h3>
            <ul class="nav-links global-nav">
                <li><a href="../week1_pytorch/index.html" class="week-link" data-week="week1_pytorch">Week 1: PyTorch</a></li>
                <li><a href="../week2_transformer/index.html" class="week-link" data-week="week2_transformer">Week 2: Transformers</a></li>
                <li><a href="../week3_llm_variants/index.html" class="week-link" data-week="week3_llm_variants">Week 3: LLMs</a></li>
                <li><a href="../week4_router/index.html" class="week-link" data-week="week4_router">Week 4: Router</a></li>
                <li><a href="../week5_diffusion/index.html" class="week-link" data-week="week5_diffusion">Week 5: Diffusion</a></li>
                <li><a href="../week6_unet/index.html" class="week-link" data-week="week6_unet">Week 6: U-Net</a></li>
                <li><a href="../week7_ldm/index.html" class="week-link" data-week="week7_ldm">Week 7: Latent Diff.</a></li>
                <li><a href="../week8_capstone/index.html" class="week-link" data-week="week8_capstone">Week 8: Capstone</a></li>
                <li><a href="../week9_lora/index.html" class="week-link" data-week="week9_lora">Week 9: LoRA</a></li>
                <li><a href="../week10_moe/index.html" class="week-link" data-week="week10_moe">Week 10: MoE</a></li>
                <li><a href="../week11_opt/index.html" class="week-link" data-week="week11_opt">Week 11: Opt</a></li>
                <li><a href="../week12_capstone/index.html" class="week-link" data-week="week12_capstone">Week 12: Final</a></li>
            </ul>
        </div>
    </nav>

    <!-- Main Content -->
    <main class="content">
        <!-- Hero -->
        <section id="intro" class="hero-section">
            <h1 class="gradient-text">{title}</h1>
            <p class="subtitle">{subtitle}</p>
            <div class="content-card">
                <h3>Overview</h3>
                {overview_html}
                <div class="checkpoint">
                    <input type="checkbox" id="check-intro" class="checkpoint-input" onchange="updateProgress()">
                    <label for="check-intro">I understand the goals of this week</label>
                </div>
            </div>
        </section>

        <!-- Theory -->
        <section id="theory" class="section">
            <h2 class="section-title">Theory & Concepts</h2>
            {theory_html}
             <div class="content-card">
                 <h3>Concept Check</h3>
                 <p>Make sure you grasp the core equations before moving to code.</p>
                 <div class="checkpoint">
                    <input type="checkbox" id="check-theory" class="checkpoint-input" onchange="updateProgress()">
                    <label for="check-theory">I have reviewed the theoretical foundations</label>
                </div>
             </div>
        </section>

        <!-- Interactive -->
        <section id="interactive" class="section">
            <h2 class="section-title">Interactive Lab: {viz_title}</h2>
            <div class="content-card viz-card">
                <p>{viz_description}</p>
                <div class="viz-container" id="viz-container">
                    {viz_html}
                </div>
                <div class="info-box"><strong>💡 Experiment:</strong> Try different values to see how the system reacts in real-time.</div>
            </div>
        </section>

        <!-- Implementation -->
        <section id="code" class="section">
            <h2 class="section-title">Code Implementation</h2>
            <div class="content-card">
                <h3>Core Logic</h3>
                <p>Below is the critical implementation from <code>project.py</code>. Analyze how the theory transforms into PyTorch code.</p>
                <div class="code-block">
                    <div class="code-header">Python <button class="copy-btn" onclick="copyCode()">Copy</button></div>
                    <pre><code id="main-code">{code_snippet}</code></pre>
                </div>
                 <div class="checkpoint">
                    <input type="checkbox" id="check-code" class="checkpoint-input" onchange="updateProgress()">
                    <label for="check-code">I understand the code implementation</label>
                </div>
            </div>
        </section>
        
        <!-- Project -->
        <section id="project" class="section">
             <h2 class="section-title">Weekly Project</h2>
             <div class="content-card">
                <h3>{project_title}</h3>
                <p>{project_desc}</p>
                <div class="project-specs">
                    <h4>Step-by-Step Implementation</h4>
                    <ul class="checklist">
                        {project_steps}
                    </ul>
                </div>
                <div class="info-box">
                    <strong>🚀 Challenge:</strong> Can you optimize this further? Check the <code>exercises.py</code> file for bonus tasks.
                </div>
                 <div class="checkpoint">
                    <input type="checkbox" id="check-project" class="checkpoint-input" onchange="updateProgress()">
                    <label for="check-project">I have completed the weekly project</label>
                </div>
             </div>
        </section>

        <footer>
            <div class="nav-buttons">
                <a href="{prev_link}" class="btn-secondary">← Previous Week</a>
                <a href="{next_link}" class="btn-primary">Next Week →</a>
            </div>
            <p style="margin-top: 40px; color: var(--text-muted);">Antigravity Learning System &copy; 2026</p>
        </footer>
    </main>

    <script src="script.js"></script>
    <script>
        // Global Progress Logic
        const CURRENT_WEEK = "{week_id}";
        
        function updateProgress() {{
            const checks = document.querySelectorAll('.checkpoint-input');
            const checked = document.querySelectorAll('.checkpoint-input:checked');
            const progress = (checked.length / checks.length) * 100;
            
            document.getElementById('page-progress').style.width = `${{progress}}%`;
            document.getElementById('progress-text').innerText = `${{Math.round(progress)}}% Complete`;
            
            // Save to localStorage
            const state = {{}};
            checks.forEach(c => state[c.id] = c.checked);
            localStorage.setItem(`progress_${{CURRENT_WEEK}}`, JSON.stringify(state));
            
            // Mark week as done if > 90%
            if (progress > 90) {{
                localStorage.setItem(`status_${{CURRENT_WEEK}}`, 'done');
            }}
        }}

        function loadProgress() {{
             const saved = JSON.parse(localStorage.getItem(`progress_${{CURRENT_WEEK}}`));
             if (saved) {{
                 Object.keys(saved).forEach(id => {{
                     const el = document.getElementById(id);
                     if(el) el.checked = saved[id];
                 }});
                 updateProgress();
             }}
             
             // Update sidebar ticks
             document.querySelectorAll('.week-link').forEach(link => {{
                 const week = link.getAttribute('data-week');
                 if (localStorage.getItem(`status_${{week}}`) === 'done') {{
                     link.classList.add('done-week');
                 }}
                 if (week === CURRENT_WEEK) link.classList.add('active-week');
             }});
        }}
        
        function copyCode() {{
            const code = document.getElementById('main-code').innerText;
            navigator.clipboard.writeText(code);
            alert('Code copied!');
        }}

        window.onload = loadProgress;
    </script>
</body>
</html>
"""

# ==========================================
# 2. CSS TEMPLATE
# ==========================================
CSS_TEMPLATE = """
:root {{
    --bg-dark: #0f172a;
    --bg-card: rgba(30, 41, 59, 0.7);
    --primary: {primary_color};
    --secondary: {secondary_color};
    --text-main: #f1f5f9;
    --muted: #94a3b8;
    --border: rgba(255, 255, 255, 0.1);
}}

* {{ box-sizing: border-box; scroll-behavior: smooth; }}

body {{
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-dark);
    color: var(--text-main);
    margin: 0;
    display: flex;
    min-height: 100vh;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 8px; }}
::-webkit-scrollbar-track {{ background: var(--bg-dark); }}
::-webkit-scrollbar-thumb {{ background: #334155; border-radius: 4px; }}
::-webkit-scrollbar-thumb:hover {{ background: #475569; }}

/* Sidebar */
.sidebar {{
    width: 280px;
    background: rgba(15, 23, 42, 0.98);
    border-right: 1px solid var(--border);
    padding: 25px;
    position: fixed;
    height: 100vh;
    overflow-y: hidden;
    display: flex;
    flex-direction: column;
    z-index: 100;
}}

.sidebar-header {{ margin-bottom: 20px; flex-shrink: 0; }}
.sidebar h2 {{ font-size: 1.4rem; margin: 0 0 10px 0; color: white; }}
.back-link {{ color: var(--muted); font-size: 0.85rem; text-decoration: none; display: block; margin-top: 10px; }}
.back-link:hover {{ color: white; }}

.nav-scroll-area {{ overflow-y: auto; flex-grow: 1; padding-right: 10px; }}
.nav-section-title {{ font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin: 20px 0 10px 0; border-bottom: 1px solid var(--border); padding-bottom: 5px; }}

.nav-links {{ list-style: none; padding: 0; margin: 0; }}
.nav-links li {{ margin-bottom: 4px; }}
.nav-links a {{
    display: block;
    padding: 8px 12px;
    color: #cbd5e1;
    text-decoration: none;
    border-radius: 6px;
    font-size: 0.9rem;
    transition: all 0.2s;
}}
.nav-links a:hover {{ background: rgba(255,255,255,0.05); color: white; }}
.active-week {{ background: rgba(255,255,255,0.1); color: var(--primary) !important; font-weight: 600; border-left: 3px solid var(--primary); }}
.done-week::after {{ content: "✓"; float: right; color: #10b981; }}

/* Progress Bar */
.progress-container {{ background: rgba(255,255,255,0.1); border-radius: 4px; height: 6px; overflow: hidden; margin-bottom: 5px; }}
.progress-fill {{ height: 100%; background: var(--primary); width: 0%; transition: width 0.5s; }}
.progress-text {{ font-size: 0.8rem; color: var(--muted); }}

/* Main Content */
.content {{
    margin-left: 280px;
    padding: 60px 80px;
    width: 100%;
    max-width: 1400px;
}}

.gradient-text {{
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3.5rem;
    margin-bottom: 1rem;
    line-height: 1.1;
}}

.subtitle {{ font-size: 1.5rem; color: var(--muted); margin-bottom: 3rem; font-weight: 300; }}

.section {{ margin-bottom: 80px; scroll-margin-top: 60px; }}
.section-title {{ font-size: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 15px; margin-bottom: 30px; color: white; }}

.content-card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 35px;
    margin-bottom: 25px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}}

h3 {{ color: var(--secondary); font-size: 1.3rem; margin-top: 0; }}
p, li {{ line-height: 1.7; color: #e2e8f0; }}

/* Code Blocks */
.code-block {{
    background: #000;
    border-radius: 8px;
    border: 1px solid var(--border);
    margin: 20px 0;
    overflow: hidden;
}}
.code-header {{ background: #1e293b; padding: 8px 15px; display: flex; justify-content: space-between; align-items: center; font-size: 0.8rem; color: #94a3b8; }}
.copy-btn {{ background: transparent; border: 1px solid #475569; color: #cbd5e1; padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem; }}
.copy-btn:hover {{ background: #334155; color: white; }}
pre {{ padding: 20px; margin: 0; overflow-x: auto; font-family: 'Fira Code', monospace; font-size: 0.9rem; color: #e2e8f0; }}

/* Interactive Viz */
.viz-container {{
    background: rgba(0,0,0,0.3);
    border: 1px solid var(--primary);
    border-radius: 12px;
    padding: 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 350px;
    margin: 20px 0;
}}

/* Info & Warning Boxes */
.info-box {{ background: rgba(59, 130, 246, 0.1); border-left: 4px solid #3b82f6; padding: 15px; margin: 20px 0; border-radius: 0 8px 8px 0; }}
.warning-box {{ background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 15px; margin: 20px 0; border-radius: 0 8px 8px 0; }}

/* Checkpoints */
.checkpoint {{ background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; display: flex; align-items: center; margin-top: 20px; }}
.checkpoint-input {{ transform: scale(1.5); margin-right: 15px; accent-color: #10b981; }}
.checkpoint label {{ cursor: pointer; font-weight: 500; color: #cbd5e1; }}

.checklist li {{ padding: 8px 0; border-bottom: 1px solid var(--border); }}

/* Navigation Buttons */
.nav-buttons {{ display: flex; justify-content: space-between; margin-top: 60px; }}
.btn-primary, .btn-secondary {{ padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; display: inline-block; transition: transform 0.2s; }}
.btn-primary {{ background: var(--primary); color: white; }}
.btn-secondary {{ background: #334155; color: white; }}
.btn-primary:hover, .btn-secondary:hover {{ transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,0,0,0.3); }}

/* Custom CSS */
{custom_css}
"""

# ==========================================
# 3. CONTENT DATA (Rich)
# ==========================================
# Note: I am writing abbreviated content for the script, but I should ensure it's dense enough.
WEEKS_DATA = {
    "week1_pytorch": {
        "title": "Week 1: PyTorch Foundations",
        "short_title": "PyTorch Basics",
        "subtitle": "Tensors, Autograd, and Neural Networks.",
        "primary_color": "#8b5cf6",
        "secondary_color": "#06b6d4",
        "overview_html": """
            <p>PyTorch is a powerful deep learning framework that provides Tensors (multi-dimensional arrays with GPU acceleration), Autograd (automatic differentiation), and Dynamic Computation Graphs.</p>
            <p>In this week, we build neural networks from first principles, avoiding high-level abstractions like <code>nn.Linear</code> until we understand the math underneath.</p>
        """,
        "theory_html": """
            <div class="content-card">
                <h3>Sensors & Operations</h3>
                <p>Tensors are the fundamental building blocks. They are similar to NumPy arrays but can run on GPUs.</p>
                <div class="code-block">x = torch.tensor([[1, 2], [3, 4]])</div>
            </div>
            <div class="content-card">
                <h3>Autograd: The Engine of Deep Learning</h3>
                <p>PyTorch automatically tracks operations on tensors to build a computational graph. When you call <code>.backward()</code>, it computes gradients using the Chain Rule.</p>
            </div>
            <div class="content-card">
                <h3>The Chain Rule</h3>
                <p>∂y/∂x = (∂y/∂h) * (∂h/∂x)</p>
                <p>Gradients flow backwards from the output loss to the input parameters.</p>
            </div>
        """,
        "viz_title": "Computational Graph",
        "viz_description": "A visualization of how data flows forward (blue) and gradients flow backward (red).",
        "viz_html": """
            <svg width="100%" height="300" viewBox="0 0 600 300">
                <g class="node" transform="translate(100,150)">
                    <circle r="30" fill="#a855f7" opacity="0.2" />
                    <circle r="25" fill="#a855f7" />
                    <text dy="5" text-anchor="middle" fill="white">x</text>
                    <text dy="45" text-anchor="middle" fill="#a855f7" font-size="12">Input</text>
                </g>
                <line x1="130" y1="150" x2="220" y2="150" stroke="#06b6d4" stroke-width="2" />
                <g class="node" transform="translate(250,150)">
                    <circle r="30" fill="#06b6d4" opacity="0.2" />
                    <circle r="25" fill="#06b6d4" />
                    <text dy="5" text-anchor="middle" fill="white">x²</text>
                    <text dy="45" text-anchor="middle" fill="#06b6d4" font-size="12">Op</text>
                </g>
                <line x1="280" y1="150" x2="370" y2="150" stroke="#10b981" stroke-width="2" />
                <g class="node" transform="translate(400,150)">
                    <circle r="30" fill="#10b981" opacity="0.2" />
                    <circle r="25" fill="#10b981" />
                    <text dy="5" text-anchor="middle" fill="white">+3</text>
                </g>
                <line x1="430" y1="150" x2="470" y2="150" stroke="#f59e0b" stroke-width="2" />
                <g class="node" transform="translate(500,150)">
                    <circle r="30" fill="#f59e0b" opacity="0.2" />
                    <circle r="25" fill="#f59e0b" />
                    <text dy="5" text-anchor="middle" fill="white">y</text>
                </g>
                <path d="M 470 140 L 430 140" stroke="#ef4444" stroke-width="3" stroke-dasharray="5,5" />
                <path d="M 370 140 L 280 140" stroke="#ef4444" stroke-width="3" stroke-dasharray="5,5" />
                <path d="M 220 140 L 130 140" stroke="#ef4444" stroke-width="3" stroke-dasharray="5,5" />
            </svg>
        """,
        "code_snippet": """class ManualLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        # Xavier initialization
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * (2.0 / in_features) ** 0.5
        )
        self.bias = nn.Parameter(torch.zeros(out_features))
    
    def forward(self, x):
        # x: (batch_size, in_features)
        # weight: (out_features, in_features)
        return x @ self.weight.T + self.bias""",
        "project_title": "MLP from Scratch",
        "project_desc": "Build a Multi-Layer Perceptron to classify MNIST digits without using nn.Linear.",
        "project_steps": """
            <li>Implement <code>ManualLinear</code> using <code>nn.Parameter</code>.</li>
            <li>Build an Architecture: 784 → 256 → 128 → 10.</li>
            <li>Implement the Training Loop with <code>optimizer.zero_grad()</code> and <code>loss.backward()</code>.</li>
            <li>Achieve >95% accuracy on MNIST.</li>
        """,
        "custom_css": "",
        "script_js": "",
        "prev_link": "../../index.html",
        "next_link": "../../modules/week2_transformer/index.html"
    },

    "week2_transformer": {
        "title": "Week 2: The Transformer Architecture",
        "short_title": "Transformer Arch",
        "subtitle": "Attention Is All You Need: The architecture that changed modern AI forever.",
        "primary_color": "#d946ef",
        "secondary_color": "#8b5cf6",
        "overview_html": """
            <p>Before Transformers, NLP relied on RNNs and LSTMs, which processed data sequentially (slow) and struggled with long-range dependencies. The <strong>Transformer</strong> (2017) revolutionized this by introducing <strong>Self-Attention</strong> and <strong>Parallelization</strong>.</p>
            <p>In this week, we dissect the architecture block by block: Positional Encoding, Multi-Head Attention, and Feed-Forward Networks.</p>
        """,
        "theory_html": """
            <div class="content-card">
                <h3>The Core Mechanism: Self-Attention</h3>
                <p>Self-attention allows the model to look at other words in the sentence to build a better representation of the current word. For example, in "The animal didn't cross the street because <strong>it</strong> was too tired", attention links "it" to "animal".</p>
                <h4>The Equation</h4>
                <div class="code-block">Attention(Q, K, V) = softmax( (QK^T) / √d_k ) V</div>
                <ul>
                    <li><strong>Query (Q):</strong> What we are looking for.</li>
                    <li><strong>Key (K):</strong> What the token offers.</li>
                    <li><strong>Value (V):</strong> The actual information content.</li>
                </ul>
            </div>
            <div class="content-card">
                <h3>Positional Encoding</h3>
                <p>Since the Transformer has no recurrence, it has no inherent notion of order. We must inject position information.</p>
                <div class="code-block">PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))</div>
            </div>
        """,
        "viz_title": "Multi-Head Attention",
        "viz_description": "Visualize how different heads focus on different parts of the sentence. Click 'Simulate' to see new patterns.",
        "viz_html": """<svg id="attn-viz" width="600" height="300"></svg><br><button onclick="visualizeAttention()">Simulate Attention Weights</button>""",
        "code_snippet": """class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch, seq_len, _ = x.shape
        
        # 1. Linear Projections & Split Heads
        Q = self.W_q(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # 3. Concatenate & Final Linear
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.W_o(output)""",
        "project_title": "Build a Transformer Block",
        "project_desc": "You will implement a reusable Transformer Encoder Layer containing Multi-Head Attention and a FeedForward Network.",
        "project_steps": """
            <li>Implement <code>PositionalEncoding</code> class using sine/cosine functions.</li>
            <li>Implement <code>MultiHeadAttention</code> with correct tensor reshaping.</li>
            <li>Build the <code>FeedForward</code> block (Linear -> ReLU -> Linear).</li>
            <li>Assemble them into a <code>TransformerBlock</code> with LayerNorm and Residual Connections.</li>
        """,
        "custom_css": "",
        "script_js": """
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
        """,
        "prev_link": "../../modules/week1_pytorch/index.html",
        "next_link": "../../modules/week3_llm_variants/index.html"
    },
    
    "week3_llm_variants": {
        "title": "Week 3: LLM Variants & Tokenization",
        "short_title": "LLMs & GPT-2",
        "subtitle": "Understanding the data pipeline and architectures of GPT-2, LLaMA, and more.",
        "primary_color": "#3b82f6",
        "secondary_color": "#ec4899",
        "overview_html": """<p>Large Language Models are not just "big transformers". They rely on sophisticated tokenization, positional embeddings (RoPE, ALiBi), and decoding strategies. This week focues on the <strong>input pipeline</strong> and the <strong>decoder-only</strong> architecture typical of GPT models.</p>""",
        "theory_html": """
        <div class="content-card">
            <h3>Byte-Pair Encoding (BPE)</h3>
            <p>Computers don't read words. They deal with numbers. Simple character-level selection is too verbose, and word-level is too sparse.</p>
            <p><strong>BPE</strong> starts with characters and iteratively merges the most frequent pairs. <code>('u', 'n') -> 'un'</code>. Eventually <code>('un', 'related') -> 'unrelated'</code>.</p>
        </div>
        <div class="content-card">
            <h3>Decoder-Only Architecture (GPT)</h3>
            <p>Unlike BERT (Encoder), GPT is a specific type of Transformer that uses <strong>Masked Self-Attention</strong>. It cannot look at future tokens.</p>
            <div class="code-block">mask = torch.tril(torch.ones(seq_len, seq_len))</div>
        </div>
        """,
        "viz_title": "Visual Tokenizer",
        "viz_description": "See how text gets chopped into tokens. Notice how common words are single tokens, while rare words are split.",
        "viz_html": """<input type="text" id="t-in" placeholder="Type text..." onkeyup="tokenize()" style="width:100%; padding:10px;"><div id="t-out" style="display:flex; flex-wrap:wrap; gap:5px; margin-top:20px;"></div>""",
        "code_snippet": """# Generating text with a causal mask
def generate(model, context, max_new_tokens):
    for _ in range(max_new_tokens):
        # Crop context to context_window
        context_cond = context[:, -block_size:]
        
        # Get predictions
        logits, _ = model(context_cond)
        
        # Focus only on the last time step
        logits = logits[:, -1, :]
        
        # Apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)
        
        # Append to running sequence
        context = torch.cat((context, idx_next), dim=1)
    return context""",
        "project_title": "Mini-GPT",
        "project_desc": "Load GPT-2 weights and implement a text generation loop with Temperature and Top-K sampling.",
        "project_steps": """
            <li>Load <code>gpt2</code> using HuggingFace Transformers.</li>
            <li>Explore the <code>vocab.json</code> to understand BPE.</li>
            <li>Implement the <code>generate()</code> function manually (no <code>model.generate</code>).</li>
            <li>Add <strong>Temperature</strong> scaling to control randomness.</li>
        """,
        "custom_css": ".token { padding: 4px 8px; border-radius: 4px; border: 1px solid rgba(255,255,255,0.2); }",
        "script_js": """
        function tokenize() {
            const text = document.getElementById('t-in').value;
            const container = document.getElementById('t-out');
            container.innerHTML = '';
            // Mock tokenizer logic
            const words = text.split(/([\\s\\S])/);
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
        """,
        "prev_link": "../../modules/week2_transformer/index.html",
        "next_link": "../../modules/week4_router/index.html"
    },
    
    "week4_router": {
        "title": "Week 4 Capstone: The Router",
        "short_title": "Router Capstone",
        "subtitle": "Building the brain of a Mixture-of-Experts (MoE) system.",
        "primary_color": "#8b5cf6",
        "secondary_color": "#10b981",
        "overview_html": "<p>A Router is a classifier that decides <em>which</em> model should handle a given query. In this capstone, you will build a system that can distinguish between Chat, Coding, and Math queries, dispatching them to simulated expert models.</p>",
        "theory_html": """
        <div class="content-card">
            <h3>Embeddings for Classification</h3>
            <p>We use a pre-trained BERT model to extract semantic features from the input text.</p>
            <p>The <code>[CLS]</code> token output (vector size 768) serves as a summary of the entire sentence.</p>
        </div>
        <div class="content-card">
            <h3>The Routing Layer</h3>
            <p>A simple linear layer projects the 768-dim vector to N_EXPERTS logits.</p>
            <div class="code-block">logits = Linear(768, 3) # [Chat, Code, Math]</div>
            <p>We train this using Cross Entropy Loss on a synthetic dataset of queries.</p>
        </div>
        """,
        "viz_title": "Live Router",
        "viz_description": "Type a query. The system will categorize it and route it to the correct simulated expert.",
        "viz_html": """
            <input type="text" id="q" placeholder="Type: 'Write a python function' or 'What is 5+5?'" style="width:100%; padding:10px; margin-bottom:10px;">
            <button onclick="route()">Route Query</button>
            <div style="display:flex; justify-content:space-around; margin-top:30px;">
                <div id="e1" class="exp chat">Other</div>
                <div id="e2" class="exp code">Code</div>
                <div id="e3" class="exp math">Math</div>
            </div>
        """,
        "code_snippet": """class Router(nn.Module):
    def __init__(self, num_experts=3):
        super().__init__()
        self.bert = AutoModel.from_pretrained('prajjwal1/bert-tiny')
        self.classifier = nn.Linear(128, num_experts) # TinyBERT dim is 128
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # Use [CLS] token embedding
        cls_emb = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_emb)
        return logits
        
    def route(self, text):
        inputs = tokenizer(text, return_tensors='pt')
        logits = self(inputs['input_ids'], inputs['attention_mask'])
        expert_id = torch.argmax(logits, dim=-1)
        return expert_id""",
        "project_title": "Build the Router",
        "project_desc": "Train a BERT-based classifier on a custom dataset.",
        "project_steps": """
            <li>Generate a synthetic dataset (queries labeled 0, 1, 2).</li>
            <li>Define the <code>Router</code> class using TinyBERT.</li>
            <li>Train for 5 epochs using AdamW.</li>
            <li>Implement the <code>dispatch()</code> function that calls fake experts.</li>
        """,
        "custom_css": ".exp { width: 80px; height: 80px; border-radius: 50%; display: flex; align-items: center; justify-content: center; background:#333; opacity:0.3; transition:0.3s; } .chat{border:2px solid blue;} .code{border:2px solid green;} .math{border:2px solid orange;} .active-e { opacity:1; transform:scale(1.2); background: #eee; color:#000; box-shadow: 0 0 15px white; }",
        "script_js": """
        function route() {
            const t = document.getElementById('q').value.toLowerCase();
            document.querySelectorAll('.exp').forEach(e=>e.classList.remove('active-e'));
            
            let id = 'e1';
            if (t.includes('code') || t.includes('function') || t.includes('print')) id='e2';
            else if (t.match(/[0-9]/)) id='e3';
            
            document.getElementById(id).classList.add('active-e');
        }
        """,
        "prev_link": "../../modules/week3_llm_variants/index.html",
        "next_link": "../../modules/week5_diffusion/index.html"
    },

    "week5_diffusion": {
        "title": "Week 5: Diffusion Math",
        "short_title": "Diffusion Basics",
        "subtitle": "From Noise to Clarity: Understanding the Forward and Reverse Processes.",
        "primary_color": "#06b6d4",
        "secondary_color": "#f43f5e",
        "overview_html": "<p>Diffusion models (like DALL-E 2 and Stable Diffusion) work by destroying data with noise and learning to repair it. This week covers the fundamental probability theory and the <strong>Reparameterization Trick</strong>.</p>",
        "theory_html": """
        <div class="content-card">
            <h3>The Forward Process (q)</h3>
            <p>We gradually add Gaussian noise. The amount of noise at step t is controlled by a variance schedule β_t.</p>
            <div class="code-block">x_t = √(1 - β_t) * x_{t-1} + √(β_t) * ε</div>
            <p>Crucially, we can jump to step t in one go:</p>
            <div class="code-block">x_t = √(α_bar) * x_0 + √(1 - α_bar) * ε</div>
        </div>
        <div class="content-card">
            <h3>The Reverse Process (p)</h3>
            <p>We train a neural network to predict the noise <code>ε</code> that was added. If we can subtract the noise, we recover the image.</p>
        </div>
        """,
        "viz_title": "Noise Schedule",
        "viz_description": "Slide t from 0 to 1000 to see how the signal is destroyed.",
        "viz_html": """<input type="range" min="0" max="100" value="0" oninput="draw(this.value)" style="width:100%"><canvas id="c" width="200" height="200" style="background:#fff; margin-top:20px; border-radius:8px;"></canvas>""",
        "code_snippet": """def q_sample(self, x_start, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
        
    sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
    sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
    
    return sqrt_alpha_cumprod_t * x_start + \
           sqrt_one_minus_alpha_cumprod_t * noise""",
        "project_title": "Diffusion Scheduler",
        "project_desc": "Implement the noise schedules and samplers.",
        "project_steps": """
            <li>Implement linear beta schedule.</li>
            <li>Pre-compute alpha bars.</li>
            <li>Create the <code>q_sample</code> function (Forward process).</li>
            <li>Visualize the degradation of an image over timesteps.</li>
        """,
        "custom_css": "",
        "script_js": """
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
        """,
        "prev_link": "../../modules/week4_router/index.html",
        "next_link": "../../modules/week6_unet/index.html"
    },
    
    "week6_unet": {
        "title": "Week 6: The U-Net",
        "short_title": "U-Net Arch",
        "subtitle": "The backbone of modern generative image models.",
        "primary_color": "#10b981", 
        "secondary_color": "#3b82f6",
        "overview_html": "<p>The U-Net architecture, originally designed for medical segmentation, is perfect for diffusion because it preserves spatial size. It uses downsampling to capture context and upsampling to localize details, linked by <strong>Skip Connections</strong>.</p>",
        "theory_html": """
        <div class="content-card">
            <h3>Architecture Shape</h3>
            <p>The "U" shape comes from:</p>
            <ul>
                <li><strong>Encoder (Down):</strong> Convolutions + MaxPool. Increases channels, decreases size.</li>
                <li><strong>Bottleneck:</strong> Deep semantic features.</li>
                <li><strong>Decoder (Up):</strong> Transpose Convolutions. Decreases channels, increases size.</li>
            </ul>
        </div>
        <div class="content-card">
            <h3>Skip Connections</h3>
            <p>We concatenate feature maps from the encoder directly to the decoder. This allows gradients to flow easily and provides high-res details to the upsampling path.</p>
        </div>
        """,
        "viz_title": "Skip Connections",
        "viz_description": "Hover over the encoder blocks (Left) to see where the information flows in the decoder (Right).",
        "viz_html": """
        <svg viewBox="0 0 400 300" style="width:100%; max-width:500px;">
            <!-- Encoder -->
            <rect x="50" y="50" width="40" height="40" fill="#10b981" class="enc-block" onmouseover="highlight(1)" onmouseout="reset()"/>
            <rect x="70" y="110" width="40" height="40" fill="#10b981" opacity="0.8" class="enc-block" onmouseover="highlight(2)" onmouseout="reset()"/>
            <rect x="90" y="170" width="40" height="40" fill="#10b981" opacity="0.6" class="enc-block" onmouseover="highlight(3)" onmouseout="reset()"/>
            
            <!-- Bottleneck -->
            <rect x="150" y="230" width="100" height="40" fill="#3b82f6"/>
            
            <!-- Decoder -->
            <rect x="270" y="170" width="40" height="40" fill="#f59e0b" opacity="0.6" class="dec-block" id="dec-3"/>
            <rect x="290" y="110" width="40" height="40" fill="#f59e0b" opacity="0.8" class="dec-block" id="dec-2"/>
            <rect x="310" y="50" width="40" height="40" fill="#f59e0b" class="dec-block" id="dec-1"/>
            
            <!-- Skip Lines -->
            <line x1="90" y1="70" x2="310" y2="70" stroke="#fff" stroke-width="2" stroke-dasharray="5,5" opacity="0.3" id="skip-1"/>
            <line x1="110" y1="130" x2="290" y2="130" stroke="#fff" stroke-width="2" stroke-dasharray="5,5" opacity="0.3" id="skip-2"/>
            <line x1="130" y1="190" x2="270" y2="190" stroke="#fff" stroke-width="2" stroke-dasharray="5,5" opacity="0.3" id="skip-3"/>
        </svg>
        """,
        "code_snippet": """class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down1 = DoubleConv(in_channels, 64)
        self.down2 = DoubleConv(64, 128)
        self.bot = DoubleConv(128, 256)
        self.up1 = Up(256, 128)
        self.up2 = Up(128, 64)
        self.out = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1) # Maxpool inside
        bot = self.bot(x2)
        x = self.up1(bot, x2) # Skip connection x2
        x = self.up2(x, x1)   # Skip connection x1
        return self.out(x)""",
        "project_title": "Build a Simple U-Net",
        "project_desc": "Construct a functional U-Net for noise prediction.",
        "project_steps": """
            <li>Implement the <code>DoubleConv</code> block.</li>
            <li>Implement <code>Down</code> (Pool) and <code>Up</code> (Upsample + Concat) blocks.</li>
            <li>Verify output shapes match input shapes.</li>
            <li>(Bonus) Add Time Embeddings for Diffusion.</li>
        """,
        "custom_css": ".enc-block:hover { stroke: white; stroke-width: 2px; cursor: pointer; }",
        "script_js": """
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
        """,
        "prev_link": "../../modules/week5_diffusion/index.html",
        "next_link": "../../modules/week7_ldm/index.html"
    },

    "week7_ldm": {
        "title": "Week 7: Latent Diffusion",
        "short_title": "Latent Diffusion",
        "subtitle": "Stable Diffusion: Making it fast by working in compressed space.",
        "primary_color": "#f59e0b",
        "secondary_color": "#ef4444",
        "overview_html": "<p>Pixel-space diffusion is slow. <strong>Latent Diffusion Models (LDMs)</strong> use a Variational Autoencoder (VAE) to compress images such that a 512x512 image becomes a 64x64 latent tensor. We then diffuse this small tensor.</p>",
        "theory_html": """
        <div class="content-card">
            <h3>Perceptual Compression</h3>
            <p>The VAE removes high-frequency details that aren't perceptually important, while keeping the semantic content.</p>
        </div>
        <div class="content-card">
            <h3>Conditioning</h3>
            <p>How do we guide generation? We use <strong>Cross-Attention</strong> in the U-Net. Text embeddings (from CLIP) are injected into the U-Net layers to steer the denoising.</p>
        </div>
        """,
        "viz_title": "Latent Compression",
        "viz_description": "Slide to compress the image into Latent Space and back. Notice the loss of detail.",
        "viz_html": """
        <div style="display:flex; justify-content:center; align-items:center;">
             <div id="pixel-view" style="width:100px; height:100px; background:linear-gradient(45deg, blue, red); display:grid; grid-template-columns:repeat(10,1fr);">
             </div>
             <div style="font-size:2rem; margin:0 20px;">➡️</div>
             <div id="latent-view" style="width:40px; height:40px; background:#333; display:grid; grid-template-columns:repeat(4,1fr); transition:0.5s;">
             </div>
        </div>
        <input type="range" min="0" max="100" value="0" oninput="compress(this.value)" style="width:100%; margin-top:20px;">
        <p id="comp-label" style="text-align:center;">Original (Pixel Space)</p>
        """,
        "code_snippet": """# Training Loop Sketch
# 1. Encode image to latent
z = vae.encode(image).latent_dist.sample() * 0.18215

# 2. Add noise
noise = torch.randn_like(z)
z_noisy = scheduler.add_noise(z, noise, t)

# 3. Predict noise (conditioned on text)
pred_noise = unet(z_noisy, t, encoder_hidden_states=prompts)

# 4. Loss
loss = F.mse_loss(pred_noise, noise)""",
        "project_title": "Latent Pipeline Mockup",
        "project_desc": "Simulate the LDM pipeline using mock components.",
        "project_steps": """
            <li>Create the `LatentDiffusion` class.</li>
            <li>Mock the VAE encoder/decoder.</li>
            <li>Mock the CLIP text encoder.</li>
            <li>Trace the tensor shapes through the entire pipeline.</li>
        """,
        "custom_css": ".px { background: rgba(255,255,255,0.5); border:1px solid #000; }",
        "script_js": """
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
        """,
        "prev_link": "../../modules/week6_unet/index.html",
        "next_link": "../../modules/week8_capstone/index.html"
    },

    "week8_capstone": {
        "title": "Week 8 Capstone: Noisy Router",
        "short_title": "Noisy Router",
        "subtitle": "Merging Text and Image generation into one system.",
        "primary_color": "#ec4899",
        "secondary_color": "#8b5cf6",
        "overview_html": "<p>A multi-modal system. If you ask for a picture, it uses Diffusion. If you ask a question, it uses an LLM. The Router makes the decision.</p>",
        "theory_html": "<p>This integrates Week 4 (Router) with Week 7 (LDM).</p>",
        "viz_title": "Multi-Modal Switch",
        "viz_description": "Toggle inputs to see the active path.",
        "viz_html": """
        <div style="display:flex; justify-content:center; gap:20px;">
            <button onclick="activate('text')" class="btn-secondary">📝 Text Query</button>
            <button onclick="activate('image')" class="btn-secondary">🖼️ Image Request</button>
        </div>
        <div style="margin-top:20px; text-align:center;">
            <div id="path-llm" style="padding:10px; border:1px solid #333; margin:5px; transition:0.3s;">LLM Branch</div>
            <div id="path-diff" style="padding:10px; border:1px solid #333; margin:5px; transition:0.3s;">Diffusion Branch</div>
        </div>
        """,
        "code_snippet": """if 'draw' in prompt:
    return self.diffusion.generate(prompt)
else:
    return self.llm.generate(prompt)""",
        "project_title": "Multi-Modal Agent",
        "project_desc": "Combine your modules.",
        "project_steps": "<li>Import Week 4 Router.</li><li>Import Week 7 LDM.</li><li>Create <code>NoisyRouter</code> class.</li>",
        "custom_css": ".active-branch { background: #10b981; color: black; transform: scale(1.05); }",
        "script_js": """
        function activate(type) {
            document.getElementById('path-llm').className = '';
            document.getElementById('path-diff').className = '';
            
            if(type === 'text') document.getElementById('path-llm').className = 'active-branch';
            else document.getElementById('path-diff').className = 'active-branch';
        }
        """,
        "prev_link": "../../modules/week7_ldm/index.html",
        "next_link": "../../modules/week9_lora/index.html"
    },

    "week9_lora": {
        "title": "Week 9: LoRA (Low-Rank Adaptation)",
        "short_title": "LoRA",
        "subtitle": "Fine-tuning massive models on consumer GPUs.",
        "primary_color": "#ef4444",
        "secondary_color": "#f59e0b",
        "overview_html": "<p>Full fine-tuning updates all weights. LoRA freezes the main weights and trains only tiny adapter matrices <code>A</code> and <code>B</code>. <code>W' = W + BA</code>.</p>",
        "theory_html": """<div class="content-card"><h3>Rank Decomposition</h3><p>If W is 1000x1000, it has 1M params. If we use rank r=4, A is 4x1000 and B is 1000x4. Total = 8000 params. That's a 99% reduction!</p></div>""",
        "viz_title": "Matrix Decomposition",
        "viz_description": "Adjust the Rank 'r' to see how the adapter matrices change size relative to the fixed weight W.",
        "viz_html": """
        <div style="display:flex; align-items:center; justify-content:center; gap:10px; font-family:monospace;">
            <div style="width:100px; height:100px; border:2px solid #fff; display:flex; align-items:center; justify-content:center;">W (Fixed)</div>
            <div>+</div>
            <div style="display:flex; flex-direction:column; align-items:center;">
                <div id="mat-b" style="width:20px; height:100px; border:2px solid #ef4444; margin-bottom:5px;"></div>
                B
            </div>
            <div>x</div>
            <div style="display:flex; flex-direction:column; align-items:center;">
                <div id="mat-a" style="width:100px; height:20px; border:2px solid #ef4444;"></div>
                A
            </div>
        </div>
        <input type="range" min="1" max="50" value="10" oninput="updateRank(this.value)" style="width:100%; margin-top:30px;">
        <p style="text-align:center;">Rank r = <span id="r-val">10</span></p>
        """,
        "code_snippet": """class LoRALinear(nn.Module):
    def __init__(self, in_ft, out_ft, r=4):
        super().__init__()
        self.linear = nn.Linear(in_ft, out_ft) # Frozen
        self.lora_A = nn.Parameter(torch.randn(r, in_ft))
        self.lora_B = nn.Parameter(torch.zeros(out_ft, r))
        
    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T)""",
        "project_title": "Implement LoRA",
        "project_desc": "Create a wrapper that converts Linear layers to LoRA layers.",
        "project_steps": "<li>Build <code>LoRALinear</code>.</li><li>Replace layers in a dummy model.</li><li>Verify only A and B have grads.</li>",
        "custom_css": "",
        "script_js": """
        function updateRank(r) {
            document.getElementById('r-val').innerText = r;
            document.getElementById('mat-b').style.width = (r*2) + 'px';
            document.getElementById('mat-a').style.height = (r*2) + 'px';
        }
        """,
        "prev_link": "../../modules/week8_capstone/index.html",
        "next_link": "../../modules/week10_moe/index.html"
    },

    "week10_moe": {
        "title": "Week 10: Mixture of Experts",
        "short_title": "MoE",
        "subtitle": "Sparse models that scale to trillions of parameters.",
        "primary_color": "#3b82f6", 
        "secondary_color": "#10b981",
        "overview_html": "<p>In an MoE, different parts of the network specialize. For each token, a Gating Network chooses the Top-K experts to process it.</p>",
        "theory_html": "<p>Formula: y = Sum( Gate(x)_i * Expert_i(x) )</p>",
        "viz_title": "Token Routing",
        "viz_description": "Watch how tokens are routed to specific experts based on their color.",
        "viz_html": """
        <canvas id="moe-canvas" width="600" height="200" style="background:#000; border-radius:8px;"></canvas>
        <button onclick="spawnToken()" style="margin-top:10px;">Spawn Token</button>
        """,
        "code_snippet": """gates = self.gate(x)
top_k_weights, top_k_indices = torch.topk(gates, k=2)""",
        "project_title": "MoE Layer",
        "project_desc": "Implement a sparse MoE layer from scratch.",
        "project_steps": "<li>Gating Net.</li><li>Sparse Dispatch.</li><li>Load Balancing Loss.</li>",
        "custom_css": "",
        "script_js": """
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
        """,
        "prev_link": "../../modules/week9_lora/index.html",
        "next_link": "../../modules/week11_opt/index.html"
    },

    "week11_opt": {
        "title": "Week 11: Optimization & Quantization",
        "short_title": "Optimization",
        "subtitle": "Making models run fast.",
        "primary_color": "#10b981", 
        "secondary_color": "#f59e0b",
        "overview_html": "<p>Quantization reduces numbers from FP32 (32-bit) to INT8 (8-bit), reducing memory by 4x and speeding up math.</p>",
        "theory_html": "<p>Dynamic vs Static Quantization.</p>",
        "viz_title": "Quantization Effect",
        "viz_description": "Toggle quantization to see how color depth (memory) decreases, while the image usually remains recognizable.",
        "viz_html": """
        <div id="q-box" style="width:200px; height:200px; background:conic-gradient(red, blue, green, yellow, red); border-radius:50%; transition:0.5s;"></div>
        <button onclick="toggleQ()" style="margin-top:20px;" id="q-btn">Apply INT8 Quantization</button>
        """,
        "code_snippet": "quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)",
        "project_title": "Benchmark",
        "project_desc": "Quantize BERT and time it.",
        "project_steps": "<li>Measure Inference Time (FP32).</li><li>Quantize.</li><li>Measure again.</li>",
        "custom_css": "",
        "script_js": """
        let q = false;
        function toggleQ() {
            q = !q;
            const b = document.getElementById('q-box');
            if(q) {
                b.style.filter = 'contrast(500%) grayscale(100%)'; // Mock quantization artifacts
                document.getElementById('q-btn').innerText = "Revert to FP32";
            } else {
                b.style.filter = 'none';
                document.getElementById('q-btn').innerText = "Apply INT8 Quantization";
            }
        }
        """,
        "prev_link": "../../modules/week10_moe/index.html",
        "next_link": "../../modules/week12_capstone/index.html"
    },

    "week12_capstone": {
        "title": "Week 12 Capstone: The Ultimate Assistant",
        "short_title": "Final Capstone",
        "subtitle": "The culmination of 12 weeks of Deep Learning.",
        "primary_color": "#8b5cf6", 
        "secondary_color": "#d946ef",
        "overview_html": "<p>Combines MoE backbone, LoRA adapters for skills, and Router for dispatch.</p>",
        "theory_html": "<p>System Architecture definition.</p>",
        "viz_title": "System Status",
        "viz_description": "Use the controls to check the pulse of your final AI Agent.",
        "viz_html": """
        <div id="sys-stat" style="width:100px; height:100px; background:#8b5cf6; border-radius:50%; box-shadow: 0 0 20px #8b5cf6; margin: 0 auto; transition:0.5s;"></div>
        <h3 style="text-align:center; transform:translateY(-70px); color:white;">ONLINE</h3>
        <button onclick="ping()" style="display:block; margin:20px auto;">Ping System</button>
        """,
        "code_snippet": "class UltimateAssistant(nn.Module): ...",
        "project_title": "Final Project",
        "project_desc": "Build the Ultimate Assistant.",
        "project_steps": "<li>Assemble all components.</li><li>Train on diverse tasks.</li><li>Serve via API.</li>",
        "custom_css": "",
        "script_js": """
        function ping() {
            const s = document.getElementById('sys-stat');
            s.style.transform = 'scale(1.5)';
            s.style.boxShadow = '0 0 50px white';
            setTimeout(() => {
                s.style.transform = 'scale(1)';
                s.style.boxShadow = '0 0 20px #8b5cf6';
            }, 300);
        }
        """,
        "prev_link": "../../modules/week11_opt/index.html",
        "next_link": "../../index.html"
    }
}

def main():
    for week, data in WEEKS_DATA.items():
        week_dir = os.path.join(BASE_DIR, week)
        if not os.path.exists(week_dir): os.makedirs(week_dir)
            
        print(f"Generating Premium {week}...")
        
        # 1. HTML
        html = HTML_TEMPLATE.format(
            week_id=week,
            title=data['title'],
            short_title=data['short_title'],
            subtitle=data['subtitle'],
            overview_html=data['overview_html'],
            theory_html=data['theory_html'],
            viz_title=data['viz_title'],
            viz_description=data['viz_description'],
            viz_html=data['viz_html'],
            code_snippet=data['code_snippet'],
            project_title=data['project_title'],
            project_desc=data['project_desc'],
            project_steps=data['project_steps'],
            prev_link=data['prev_link'],
            next_link=data['next_link']
        )
        with open(os.path.join(week_dir, "index.html"), "w", encoding='utf-8') as f:
            f.write(html)
            
        # 2. CSS
        css = CSS_TEMPLATE.format(
            primary_color=data['primary_color'],
            secondary_color=data['secondary_color'],
            custom_css=data['custom_css']
        )
        with open(os.path.join(week_dir, "style.css"), "w", encoding='utf-8') as f:
            f.write(css)
            
        # 3. JS
        with open(os.path.join(week_dir, "script.js"), "w", encoding='utf-8') as f:
            f.write(data['script_js'])
            
if __name__ == "__main__":
    main()
