// ===================================
// Interactive Features for PyTorch Tutorial
// ===================================

document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initCopyButtons();
    initCheckpoints();
    initVisualizations();
    initScrollSpy();
});

// ===================================
// Navigation & Scroll Spy
// ===================================
function initNavigation() {
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                targetSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });
}

function initScrollSpy() {
    const sections = document.querySelectorAll('.section');
    const navLinks = document.querySelectorAll('.nav-links a');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const id = entry.target.getAttribute('id');
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${id}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }, { threshold: 0.3 });
    
    sections.forEach(section => observer.observe(section));
}

// ===================================
// Copy to Clipboard
// ===================================
function initCopyButtons() {
    const copyButtons = document.querySelectorAll('.copy-btn');
    
    copyButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const codeId = button.getAttribute('data-copy');
            const codeElement = document.getElementById(codeId);
            
            if (codeElement) {
                const code = codeElement.textContent;
                
                try {
                    await navigator.clipboard.writeText(code);
                    
                    // Visual feedback
                    const originalText = button.textContent;
                    button.textContent = 'Copied!';
                    button.classList.add('copied');
                    
                    setTimeout(() => {
                        button.textContent = originalText;
                        button.classList.remove('copied');
                    }, 2000);
                } catch (err) {
                    console.error('Failed to copy:', err);
                    button.textContent = 'Failed';
                    setTimeout(() => {
                        button.textContent = 'Copy';
                    }, 2000);
                }
            }
        });
    });
}

// ===================================
// Checkpoint Progress Tracking
// ===================================
function initCheckpoints() {
    const checkboxes = document.querySelectorAll('.checkpoint-input');
    const progressRing = document.querySelector('.progress-ring-circle');
    const progressText = document.querySelector('.progress-text');
    
    // Load saved progress from localStorage
    loadProgress();
    
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', () => {
            saveProgress();
            updateProgressRing();
        });
    });
    
    function saveProgress() {
        const progress = {};
        checkboxes.forEach(checkbox => {
            progress[checkbox.id] = checkbox.checked;
        });
        localStorage.setItem('pytorch-tutorial-progress', JSON.stringify(progress));
    }
    
    function loadProgress() {
        const saved = localStorage.getItem('pytorch-tutorial-progress');
        if (saved) {
            const progress = JSON.parse(saved);
            checkboxes.forEach(checkbox => {
                if (progress[checkbox.id]) {
                    checkbox.checked = true;
                }
            });
        }
        updateProgressRing();
    }
    
    function updateProgressRing() {
        const total = checkboxes.length;
        const checked = document.querySelectorAll('.checkpoint-input:checked').length;
        const percentage = Math.round((checked / total) * 100);
        
        // Update ring (circumference = 2πr = 2π(26) ≈ 163)
        const circumference = 163;
        const offset = circumference - (percentage / 100) * circumference;
        
        if (progressRing) {
            progressRing.style.strokeDashoffset = offset;
        }
        
        if (progressText) {
            progressText.textContent = `${percentage}%`;
        }
    }
}

// ===================================
// Interactive Visualizations
// ===================================
function initVisualizations() {
    initAutogradViz();
    initLossLandscape();
}

// Autograd Computational Graph Animation
function initAutogradViz() {
    const animateBtn = document.getElementById('animate-autograd');
    
    if (animateBtn) {
        animateBtn.addEventListener('click', () => {
            const nodes = document.querySelectorAll('#autograd-viz .node');
            const gradientFlow = document.querySelector('.gradient-flow');
            
            // Reset animation
            nodes.forEach(node => {
                node.style.opacity = '0.3';
            });
            gradientFlow.style.opacity = '0';
            
            // Forward pass animation
            nodes.forEach((node, index) => {
                setTimeout(() => {
                    node.style.opacity = '1';
                    node.style.transform = 'scale(1.1)';
                    setTimeout(() => {
                        node.style.transform = 'scale(1)';
                    }, 300);
                }, index * 500);
            });
            
            // Backward pass animation
            setTimeout(() => {
                gradientFlow.style.opacity = '1';
                
                // Animate nodes in reverse
                for (let i = nodes.length - 1; i >= 0; i--) {
                    setTimeout(() => {
                        nodes[i].style.transform = 'scale(1.1)';
                        setTimeout(() => {
                            nodes[i].style.transform = 'scale(1)';
                        }, 300);
                    }, (nodes.length - 1 - i) * 500);
                }
            }, nodes.length * 500 + 500);
            
            // Reset after animation
            setTimeout(() => {
                nodes.forEach(node => {
                    node.style.opacity = '1';
                });
            }, nodes.length * 1000 + 2000);
        });
    }
}

// Loss Landscape Visualization
function initLossLandscape() {
    const canvas = document.getElementById('loss-canvas');
    
    if (canvas) {
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        
        // Draw loss landscape (simplified 2D representation)
        for (let x = 0; x < width; x++) {
            for (let y = 0; y < height; y++) {
                // Normalize coordinates
                const nx = (x - width / 2) / 50;
                const ny = (y - height / 2) / 50;
                
                // Simple loss function: combination of quadratic and sinusoidal
                const loss = (nx * nx + ny * ny) / 10 + 
                            Math.sin(nx) * Math.cos(ny) * 2;
                
                // Map loss to color (purple gradient)
                const normalized = Math.max(0, Math.min(1, (loss + 5) / 10));
                const r = Math.floor(168 * normalized + 10);
                const g = Math.floor(85 * normalized + 10);
                const b = Math.floor(247 * normalized + 15);
                
                ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
                ctx.fillRect(x, y, 1, 1);
            }
        }
        
        // Draw gradient descent path
        drawGradientPath(ctx, width, height);
    }
}

function drawGradientPath(ctx, width, height) {
    ctx.strokeStyle = '#10b981';
    ctx.lineWidth = 3;
    ctx.beginPath();
    
    // Simulate gradient descent path
    let x = width * 0.8;
    let y = height * 0.2;
    
    ctx.moveTo(x, y);
    
    for (let i = 0; i < 50; i++) {
        // Move towards center (minimum)
        const dx = (width / 2 - x) * 0.15;
        const dy = (height / 2 - y) * 0.15;
        
        x += dx + (Math.random() - 0.5) * 5;
        y += dy + (Math.random() - 0.5) * 5;
        
        ctx.lineTo(x, y);
    }
    
    ctx.stroke();
    
    // Draw start point
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(width * 0.8, height * 0.2, 5, 0, Math.PI * 2);
    ctx.fill();
    
    // Draw end point (minimum)
    ctx.fillStyle = '#10b981';
    ctx.beginPath();
    ctx.arc(x, y, 5, 0, Math.PI * 2);
    ctx.fill();
}

// ===================================
// Smooth Scroll Enhancement
// ===================================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ===================================
// Mobile Menu Toggle (for responsive)
// ===================================
function createMobileMenuToggle() {
    if (window.innerWidth <= 1024) {
        const sidebar = document.querySelector('.sidebar');
        const content = document.querySelector('.content');
        
        // Create toggle button if it doesn't exist
        if (!document.querySelector('.menu-toggle')) {
            const toggleBtn = document.createElement('button');
            toggleBtn.className = 'menu-toggle';
            toggleBtn.innerHTML = '☰';
            toggleBtn.style.cssText = `
                position: fixed;
                top: 20px;
                left: 20px;
                z-index: 1000;
                background: var(--accent-purple);
                color: white;
                border: none;
                width: 50px;
                height: 50px;
                border-radius: 50%;
                font-size: 1.5rem;
                cursor: pointer;
                box-shadow: var(--shadow-md);
            `;
            
            toggleBtn.addEventListener('click', () => {
                sidebar.classList.toggle('open');
            });
            
            document.body.appendChild(toggleBtn);
            
            // Close sidebar when clicking outside
            content.addEventListener('click', () => {
                if (sidebar.classList.contains('open')) {
                    sidebar.classList.remove('open');
                }
            });
        }
    }
}

// Initialize mobile menu on load and resize
window.addEventListener('load', createMobileMenuToggle);
window.addEventListener('resize', createMobileMenuToggle);

// ===================================
// Keyboard Shortcuts
// ===================================
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K: Focus search (if implemented)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        // Could implement search functionality here
    }
    
    // Arrow keys for navigation
    if (e.altKey) {
        const sections = Array.from(document.querySelectorAll('.section'));
        const currentSection = sections.find(section => {
            const rect = section.getBoundingClientRect();
            return rect.top >= 0 && rect.top < window.innerHeight / 2;
        });
        
        if (currentSection) {
            const currentIndex = sections.indexOf(currentSection);
            
            if (e.key === 'ArrowDown' && currentIndex < sections.length - 1) {
                e.preventDefault();
                sections[currentIndex + 1].scrollIntoView({ behavior: 'smooth' });
            } else if (e.key === 'ArrowUp' && currentIndex > 0) {
                e.preventDefault();
                sections[currentIndex - 1].scrollIntoView({ behavior: 'smooth' });
            }
        }
    }
});

// ===================================
// Performance: Lazy Load Images (if any added)
// ===================================
if ('IntersectionObserver' in window) {
    const imageObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                if (img.dataset.src) {
                    img.src = img.dataset.src;
                    img.removeAttribute('data-src');
                    imageObserver.unobserve(img);
                }
            }
        });
    });
    
    document.querySelectorAll('img[data-src]').forEach(img => {
        imageObserver.observe(img);
    });
}

// ===================================
// Console Easter Egg
// ===================================
console.log('%c🚀 PyTorch Mastery Tutorial', 'font-size: 20px; font-weight: bold; color: #a855f7;');
console.log('%cKeep learning and building amazing things!', 'font-size: 14px; color: #06b6d4;');
console.log('%cTip: Use Alt + Arrow Keys to navigate between sections', 'font-size: 12px; color: #10b981;');
