import os
import re

# ==========================================
# 1. THE PREMIUM TEMPLATE
# ==========================================
# This is a simplified version of Week 1's index.html, adapted for templating.
TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} | Deep Learning Mastery</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Fira+Code:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg-dark: #0f172a;
            --bg-card: rgba(30, 41, 59, 0.7);
            --primary: #8b5cf6;
            --secondary: #06b6d4;
            --accent: #f59e0b;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
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
        
        /* Sidebar */
        .sidebar {{
            width: 260px;
            background: rgba(15, 23, 42, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
            padding: 20px;
            position: fixed;
            height: 100vh;
            overflow-y: auto;
            backdrop-filter: blur(10px);
            z-index: 100;
        }}
        
        .nav-links {{ list-style: none; padding: 0; }}
        .nav-links li {{ margin-bottom: 8px; }}
        .nav-links a {{
            display: block;
            padding: 10px 15px;
            color: var(--text-muted);
            text-decoration: none;
            border-radius: 8px;
            transition: all 0.2s;
            font-size: 0.95rem;
        }}
        .nav-links a:hover, .nav-links a.active {{
            background: rgba(139, 92, 246, 0.1);
            color: var(--primary);
        }}
        
        /* Main Content */
        .content {{
            margin-left: 260px;
            padding: 40px;
            width: 100%;
            max-width: 1200px;
        }}
        
        /* Typography & Cards */
        h1, h2, h3 {{ color: white; }}
        .hero-title {{ 
            font-size: 3rem; 
            background: linear-gradient(to right, #c084fc, #6366f1);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        
        .section {{ margin-bottom: 60px; scroll-margin-top: 40px; }}
        .content-card {{
            background: var(--bg-card);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 30px;
            margin-top: 20px;
            backdrop-filter: blur(10px);
        }}
        
        /* Code Blocks */
        .code-block {{
            background: #000;
            border-radius: 8px;
            padding: 15px;
            font-family: 'Fira Code', monospace;
            font-size: 0.9rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin: 15px 0;
            overflow-x: auto;
            color: #d4d4d4;
        }}
        
        /* Visualization Container */
        .viz-container {{
            background: rgba(0, 0, 0, 0.4);
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            border: 1px solid var(--primary);
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        /* Buttons & Interactive Elements */
        button {{
            background: var(--primary);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 500;
            transition: transform 0.1s;
        }}
        button:hover {{ transform: translateY(-1px); }}
        
        input, select {{
            background: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.1);
            color: white;
            padding: 10px;
            border-radius: 6px;
            margin: 5px;
        }}

        /* Custom Styles from specific Weeks */
        {custom_css}

    </style>
</head>
<body>

    <!-- Sidebar -->
    <nav class="sidebar">
        <div style="margin-bottom: 30px;">
            <h2 style="font-size: 1.2rem;">{title}</h2>
            <a href="../../index.html" style="color: var(--secondary); font-size: 0.9rem;">← Back to Dashboard</a>
        </div>
        <ul class="nav-links">
            <li><a href="#intro">📖 Introduction</a></li>
            <li><a href="#concepts">🧠 Key Concepts</a></li>
            <li><a href="#lab">🔬 Interactive Lab</a></li>
            <li><a href="#implementation">💻 Implementation</a></li>
        </ul>
    </nav>

    <!-- Main Content -->
    <main class="content">
        <!-- Hero -->
        <section id="intro">
            <h1 class="hero-title">{title}</h1>
            <p style="font-size: 1.2rem; color: var(--text-muted);">{subtitle}</p>
            
            <div class="content-card">
                <h3>Overview</h3>
                {intro_html}
            </div>
        </section>

        <!-- Concepts -->
        <section id="concepts" class="section">
            <h2 class="section-title">Key Concepts</h2>
            {concepts_html}
        </section>

        <!-- Interactive Lab -->
        <section id="lab" class="section">
            <h2 class="section-title">Interactive Visualization</h2>
            <div class="content-card">
                <p>Experiment with the concepts directly in your browser.</p>
                
                <div class="viz-container" id="viz-root">
                    {viz_html}
                </div>
            </div>
        </section>

        <!-- Implementation -->
        <section id="implementation" class="section">
            <h2 class="section-title">Implementation Details</h2>
            <div class="content-card">
                <h3>Core Logic</h3>
                <p>Here is the key implementation logic from <code>project.py</code>.</p>
                {code_html}
            </div>
        </section>
        
        <footer style="margin-top: 50px; color: var(--text-muted); text-align: center;">
            <p>Antigravity Learning System &copy; 2024</p>
        </footer>
    </main>

    <script>
        // Smooth scrolling for sidebar links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
        
        {viz_script}
    </script>
</body>
</html>
"""

# ==========================================
# 2. CONTENT DATABASE
# ==========================================

CONTENT_DB = {
    "week4_router": {
        "title": "Week 4: The Router",
        "subtitle": "Building Intelligent Dispatchers for MoE Systems",
        "custom_css": """
            .expert { padding: 15px; border-radius: 8px; width: 150px; text-align: center; opacity: 0.3; transition: all 0.3s; margin: 10px; display: inline-block; }
            .chat { background: #3b82f6; } /* Blue */
            .code { background: #10b981; } /* Green */
            .math { background: #f59e0b; } /* Orange */
            .active { opacity: 1; transform: scale(1.1); box-shadow: 0 0 15px white; }
        """,
        "intro_html": """
            <p>A <strong>Router</strong> is the decision-making brain of a modular AI system. Instead of one giant model trying to do everything, we use a router to analyze an input query and send it to the best "Expert" model for the job.</p>
            <ul>
                <li><strong>Efficiency:</strong> Only activate the parts of the brain you need.</li>
                <li><strong>Specialization:</strong> Experts can be highly specialized (e.g., Python Expert vs. Poetry Expert).</li>
            </ul>
        """,
        "concepts_html": """
            <div class="content-card">
                <h3>Embeddings & Classification</h3>
                <p>The Router works by converting text into <strong>Embeddings</strong> (vector representations) using a small model like BERT. A classifier layer then predicts which Expert ID (0, 1, 2...) should handle the request.</p>
                <div class="code-block">
                    Input Text ("Solve 2+2") → BERT → [CLS] Vector → Classifier → Class 2 (Math)
                </div>
            </div>
        """,
        "code_html": """
            <div class="code-block">
<pre>class RouterNetwork(nn.Module):
    def __init__(self, num_experts=3):
        super().__init__()
        # TinyBERT for fast embeddings
        self.bert = BertModel.from_pretrained('prajjwal1/bert-tiny')
        self.classifier = nn.Linear(128, num_experts)
        
    def forward(self, input_ids):
        # ...
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits</pre>
            </div>
        """,
        "viz_html": """
            <h3>🔀 Intelligent Router Demo</h3>
            <p>Type a query below to see how the router classifies intent.</p>
            <input type="text" id="query" placeholder="Type a request (e.g., 'Write python code')" style="width: 300px;">
            <button onclick="route()">Route Request</button>
            
            <div style="margin-top: 20px; border: 1px solid var(--primary); padding: 20px; border-radius: 10px; width: 100%; text-align: center;">
                <div id="exp-0" class="expert chat">🗣️ Chat Expert</div>
                <div id="exp-1" class="expert code">💻 Code Expert</div>
                <div id="exp-2" class="expert math">🧮 Math Expert</div>
            </div>
        """,
        "viz_script": """
            function route() {
                const query = document.getElementById('query').value.toLowerCase();
                let expertId = 0; // Default to Chat

                if (query.includes('code') || query.includes('function') || query.includes('print') || query.includes('def')) {
                    expertId = 1;
                } else if (query.includes('math') || query.includes('calculate') || query.match(/\\d+/)) {
                    expertId = 2;
                }

                // Animate
                document.querySelectorAll('.expert').forEach(e => e.classList.remove('active'));
                setTimeout(() => {
                    document.getElementById(`exp-${expertId}`).classList.add('active');
                }, 200);
            }
        """
    },
    "week5_diffusion": {
        "title": "Week 5: Diffusion Math",
        "subtitle": "Understanding the Forward & Reverse Processes",
        "custom_css": "",
        "intro_html": """
            <p>Diffusion Models work by destroying data and learning to reconstruct it. We slowly add Gaussian noise to an image until it is pure noise (<strong>Forward Process</strong>), then train a neural network to reverse this step-by-step.</p>
        """,
        "concepts_html": """
             <div class="content-card">
                <h3>The Forward Process q(x_t | x_0)</h3>
                <p>We add noise according to a <strong>Schedule</strong> (beta). As t increases, the signal x_0 fades and noise increases.</p>
                <div class="code-block">x_t = √α_bar * x_0 + √(1 - α_bar) * ε</div>
                <p>This allows us to sample x_t at any timestep directly, without iteration. This property is crucial for efficient training.</p>
            </div>
        """,
        "code_html": """
            <div class="code-block">
<pre>def q_sample(self, x_start, t, noise=None):
    # Diffuse the data (Forward Process)
    sqrt_alpha = self.sqrt_alphas_cumprod[t]
    sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t]
    
    return sqrt_alpha * x_start + sqrt_one_minus * noise</pre>
            </div>
        """,
        "viz_html": """
            <h3>Forward Diffusion Slider</h3>
            <p>Drag the slider to increase timestep 't' and add noise.</p>
            <input type="range" id="noiseSlider" min="0" max="100" value="0" style="width: 100%">
            <p>Timestep: <span id="timeVal">0</span></p>
            
            <div style="width: 200px; height: 200px; background: white; position: relative; display: flex; align-items: center; justify-content: center; overflow: hidden;" id="canvas-container">
                <div style="width: 100px; height: 100px; background: black; z-index: 1;"></div>
                <!-- Noise overlay -->
                <canvas id="noiseCanvas" width="200" height="200" style="position: absolute; top: 0; left: 0; z-index: 2; pointer-events: none;"></canvas>
            </div>
        """,
        "viz_script": """
            const slider = document.getElementById('noiseSlider');
            const timeVal = document.getElementById('timeVal');
            const canvas = document.getElementById('noiseCanvas');
            const ctx = canvas.getContext('2d');

            slider.oninput = function() {
                const t = this.value;
                timeVal.innerText = t;
                drawNoise(t);
            }

            function drawNoise(intensity) {
                ctx.clearRect(0, 0, 200, 200);
                // Create noise
                const idata = ctx.createImageData(200, 200);
                const buffer32 = new Uint32Array(idata.data.buffer);
                const len = buffer32.length;
                
                // Alpha varies with intensity (0 to 255)
                // Actually we want to replace pixels with noise
                // Simulation: Draw random pixels with opacity based on t
                
                for (let i = 0; i < len; i++) {
                    if (Math.random() * 100 < intensity) {
                        // Random static
                        const gray = Math.random() * 255;
                        // ABGR order
                        buffer32[i] = (255 << 24) | (gray << 16) | (gray << 8) | gray;
                    }
                }
                ctx.putImageData(idata, 0, 0);
            }
            drawNoise(0);
        """
    },
    "week6_unet": {
        "title": "Week 6: The U-Net",
        "subtitle": "The Backbone of Image Generation",
        "custom_css": "",
        "intro_html": "<p>The U-Net is a convolutional neural network with a unique 'U' shape. It compresses an image down to capture context ('What is in the image?') and then expands it back up to localize details ('Where is it?'), using Skip Connections to preserve fine details.</p>",
        "concepts_html": """<div class="content-card"><h3>Skip Connections</h3><p>Deep networks lose spatial information. Skip connections concatenate high-resolution features from the encoder directly to the decoder. This is why U-Nets produce such sharp images.</p></div>""",
        "code_html": "See project.py for full implementation.",
        "viz_html": """<h3>U-Net Architecture Visualizer</h3><p>Hover over layers to see how dimensions change.</p><div style="display:flex; justify-content:center; align-items:center; height: 300px; gap: 5px;">
            <div style="height: 200px; width: 40px; background: #3b82f6; display:flex; align-items:center; justify-content:center;">64</div>
            <div style="height: 150px; width: 40px; background: #3b82f6;">128</div>
            <div style="height: 100px; width: 40px; background: #3b82f6;">256</div>
            <div style="height: 50px; width: 40px; background: #ef4444;">512</div>
            <div style="height: 100px; width: 40px; background: #10b981;">256</div>
            <div style="height: 150px; width: 40px; background: #10b981;">128</div>
            <div style="height: 200px; width: 40px; background: #10b981;">64</div>
        </div>
        <p style="text-align:center"><span style="color:#3b82f6">Encoder (Down)</span> → <span style="color:#ef4444">Bottleneck</span> → <span style="color:#10b981">Decoder (Up)</span></p>""",
        "viz_script": ""
    },
     "week7_ldm": {
        "title": "Week 7: Latent Diffusion",
        "subtitle": "High-Resolution Image Synthesis with Latents",
        "custom_css": "",
        "intro_html": "<p>Standard diffusion on pixels is slow. <strong>Latent Diffusion Models (LDMs)</strong>, like Stable Diffusion, use a VAE to compress images into a small 'Latent Space' first. The diffusion process happens in this efficient space, 48x smaller than the original image.</p>",
        "concepts_html": """<div class="content-card"><h3>The VAE Trick</h3><p>Perceptual Compression: We keep the semantics, discard the imperceptible pixel noise.
        <br><strong>Pixel Space (512x512x3)</strong> ≈ 786,432 values
        <br><strong>Latent Space (64x64x4)</strong> ≈ 16,384 values (50x compression!)
        </p></div>""",
        "code_html": "See project.py...",
        "viz_html": """<h3>Latent Interpolation</h3><p>Morphing between two concepts in latent space.</p>
        <div style="display:flex; justify-content:center; gap: 20px; align-items:center;">
            <div style="text-align:center">🐶<br>Dog Vector</div>
            <input type="range" id="interp" min="0" max="100">
            <div style="text-align:center">🐱<br>Cat Vector</div>
        </div>
        <div id="result" style="text-align:center; font-size: 3rem; margin-top: 20px;">🐶</div>
        """,
        "viz_script": """
            document.getElementById('interp').oninput = function() {
                const val = this.value;
                const res = document.getElementById('result');
                if(val < 25) res.innerText = '🐶';
                else if(val < 50) res.innerText = '🐕'; // Dog-ish
                else if(val < 75) res.innerText = '🐈'; // Cat-ish
                else res.innerText = '🐱';
            }
        """
    },
     "week8_capstone": {
        "title": "Week 8: The Noisy Router",
        "subtitle": "Capstone: Merging LLMs with Diffusion",
        "custom_css": """ .expert-card { border: 1px solid white; padding: 20px; margin: 10px; border-radius: 8px; } """, 
        "intro_html": "<p>This Capstone integrates the <strong>Router</strong> (Week 4) with the <strong>Diffusion Model</strong> (Week 5-7). The system listens to the user: if they want to 'draw', it routes to the Diffusion Expert; if they want 'chat', it routes to the LLM Expert.</p>",
        "concepts_html": "",
        "code_html": "See project.py...",
        "viz_html": """<h3>System Demo</h3>
        <input type="text" id="cap8-input" placeholder="Try 'Draw a robot' or 'Explain quantum physics'" style="width: 300px;">
        <button onclick="runCapstone8()">Send</button>
        <div id="cap8-log" style="margin-top:20px; font-family:monospace; color:#10b981;"></div>
        """,
        "viz_script": """
            function runCapstone8() {
                const txt = document.getElementById('cap8-input').value.toLowerCase();
                const log = document.getElementById('cap8-log');
                log.innerHTML = `> Analyzing '${txt}'...<br>`;
                
                setTimeout(() => {
                    if(txt.includes('draw') || txt.includes('image') || txt.includes('paint')) {
                        log.innerHTML += "> Intent: GENERATION<br>> Routing to: Stable Diffusion Expert<br>> [Generating Image...] 🎨";
                    } else {
                        log.innerHTML += "> Intent: CONVERSATION<br>> Routing to: LLM Expert<br>> [Generating Text...] 💬";
                    }
                }, 800);
            }
        """
    },
     "week9_lora": {
        "title": "Week 9: LoRA",
        "subtitle": "Low-Rank Adaptation of LLMs",
        "custom_css": "",
        "intro_html": "<p>Fine-tuning huge models is expensive. <strong>LoRA</strong> freezes the original weights (W) and injects trainable low-rank matrices (A and B). This reduces trainable parameters by 10,000x!</p>",
        "concepts_html": "<div class='content-card'><h3>Matrix Decomposition</h3><p>W_new = W_frozen + B × A. <br>If W is 1000x1000 (1M params), and Rank=4:<br>A is 4x1000 (4k), B is 1000x4 (4k). Total 8k params only.</p></div>",
        "code_html": "",
        "viz_html": """
        <h3>Parameter Savings Calculator</h3>
        <p>Model Dim: <input type="number" id="dim" value="4096"></p>
        <p>LoRA Rank: <input type="number" id="rank" value="8"></p>
        <button onclick="calcLora()">Calculate</button>
        <p id="lora-res"></p>
        """,
        "viz_script": """
            function calcLora() {
                const d = parseInt(document.getElementById('dim').value);
                const r = parseInt(document.getElementById('rank').value);
                const full = d * d;
                const lora = 2 * d * r;
                const ratio = (lora/full) * 100;
                document.getElementById('lora-res').innerHTML = `
                Full Layer Params: ${full.toLocaleString()}<br>
                LoRA Params: <span style='color:#10b981'>${lora.toLocaleString()}</span><br>
                Reduction: Only <strong>${ratio.toFixed(4)}%</strong> of original size!
                `;
            }
        """
    },
    "week10_moe": {
        "title": "Week 10: Mixture of Experts",
         "subtitle": "Scaling Models with Conditional Computation",
        "custom_css": ".moe-node { width: 40px; height: 40px; border-radius: 50%; display: inline-flex; align-items:center; justify-content:center; margin: 5px; background: #334155; } .active-exp { background: #8b5cf6; box-shadow: 0 0 10px #8b5cf6; }",
        "intro_html": "<p>MoE models have huge capacity but low inference cost. They route each token to only a few 'Experts'. This allows models like GPT-4 to have trillions of parameters while only using a fraction for each word.</p>",
        "concepts_html": "",
        "code_html": "",
        "viz_html": """
        <h3>Token Routing Viz</h3>
        <div id="token-stream" style="font-size: 1.5rem; margin-bottom: 20px;">
            Input: <span id="curr-tok">...</span>
        </div>
        <div style="display:flex; flex-wrap:wrap; width: 300px;">
            <div id="ex-0" class="moe-node">E1</div>
            <div id="ex-1" class="moe-node">E2</div>
            <div id="ex-2" class="moe-node">E3</div>
            <div id="ex-3" class="moe-node">E4</div>
        </div>
        <button onclick="startMoE()">Simulate Stream</button>
        """,
        "viz_script": """
            function startMoE() {
                const tokens = ["The", "quick", "brown", "fox", "jumps", "over", "lazy", "dog"];
                let i=0;
                const loop = setInterval(() => {
                    if(i >= tokens.length) { clearInterval(loop); return; }
                    document.getElementById('curr-tok').innerText = tokens[i];
                    
                    document.querySelectorAll('.moe-node').forEach(n => n.classList.remove('active-exp'));
                    // Random expert
                    const exp = Math.floor(Math.random() * 4);
                    document.getElementById(`ex-${exp}`).classList.add('active-exp');
                    i++;
                }, 800);
            }
        """
    },
    "week11_opt": {
        "title": "Week 11: Optimization",
        "subtitle": "Quantization & Efficient Inference",
         "custom_css": "",
        "intro_html": "Making models smaller and faster using Quantization (INT8).",
        "concepts_html": "FP32 (32-bit floats) vs INT8 (8-bit integers). 4x memory savings.",
        "code_html": "",
        "viz_html": """<h3>Memory Footprint Comparison</h3>
        <div style="display:flex; align-items:flex-end; height: 200px; gap: 20px;">
            <div style="width: 100px; height: 100%; background: #ef4444; position: relative;">
                <span style="position:absolute; top:50%; width:100%; text-align:center;">FP32<br>100%</span>
            </div>
            <div style="width: 100px; height: 25%; background: #10b981; position: relative;">
                <span style="position:absolute; bottom:105%; width:100%; text-align:center;">INT8<br>25%</span>
            </div>
        </div>""",
        "viz_script": ""
    },
    "week12_capstone": {
        "title": "Week 12: Ultimate Capstone",
        "subtitle": "The LoRA-MoE Router",
         "custom_css": "",
        "intro_html": "Combining everything: MoE backbone for knowledge, LoRA adapters for skills.",
        "concepts_html": "Modular AI Architecture.",
        "code_html": "",
        "viz_html": "<h3>The Final Architecture</h3><p>Visualizing the full stack...</p><div style='border:1px solid white; padding:20px; text-align:center;'>User Input<br>↓<br>Router<br>↙ ↘<br>Coding Expert (LoRA A) &nbsp;&nbsp;&nbsp; Creative Expert (LoRA B)</div>",
        "viz_script": ""
    }
}

# ==========================================
# 3. GENERATOR LOGIC
# ==========================================
def main():
    base_dir = "interactive_platform/modules"
    
    for week_key, data in CONTENT_DB.items():
        path = os.path.join(base_dir, week_key, "interactive.html")
        
        # Verify dir exists
        if not os.path.exists(os.path.dirname(path)):
            print(f"Skipping {week_key} (dir not found)")
            continue
            
        print(f"Upgrading {week_key}...")
        
        html = TEMPLATE.format(
            title=data['title'],
            subtitle=data['subtitle'],
            custom_css=data['custom_css'],
            intro_html=data['intro_html'],
            concepts_html=data.get('concepts_html', '<p>Coming soon...</p>'),
            code_html=data.get('code_html', '<p>Check project.py</p>'),
            viz_html=data['viz_html'],
            viz_script=data['viz_script']
        )
        
        with open(path, "w", encoding='utf-8') as f:
            f.write(html)

if __name__ == "__main__":
    main()
