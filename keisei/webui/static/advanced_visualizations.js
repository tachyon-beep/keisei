// Advanced Deep Reinforcement Learning Visualizations for Keisei WebUI
class AdvancedVisualizationManager {
    constructor() {
        this.currentMode = 'technical';
        this.visibilityTiers = [1];
        this.animationCoordinator = new AnimationCoordinator();
        // Performance monitoring handled by main WebUI app.js
        
        // Visualization instances
        this.neuralHeatmap = null;
        this.explorationGauge = null;
        this.advantageChart = null;
        this.skillRadar = null;
        this.gradientFlow = null;
        this.bufferDynamics = null;
        this.tournamentTree = null;
        this.strategyRadar = null;
        
        // Data buffers for trends
        this.dataBuffers = {
            advantage: [],
            exploration: [],
            gradients: [],
            skills: { opening: [], tactics: [], endgame: [], strategy: [], pattern: [], time: [] }
        };
        
        this.init();
    }
    
    init() {
        console.log('🎨 Initializing Advanced Visualization Manager...');
        this.setupModeControls();
        this.initializeVisualizations();
        this.setupProgressiveDisclosure();
        this.setupKeyboardControls();
        this.setupResizeHandler();
        
        // Ensure proper visibility for current mode
        this.adjustVisualizationVisibility();
        
        console.log('✅ Advanced Visualization Manager initialized');
        console.log('🎯 Available update methods:', [
            'updateNeuralConfidence', 'updateExploration', 'updateAdvantage', 
            'updateSkills', 'updateGradients', 'updateBuffer'
        ]);
    }
    
    setupModeControls() {
        const modeButtons = document.querySelectorAll('.mode-btn');
        modeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                const mode = e.target.dataset.mode;
                this.switchMode(mode);
            });
        });
    }
    
    switchMode(mode) {
        const dashboard = document.querySelector('.dashboard');
        const buttons = document.querySelectorAll('.mode-btn');
        
        // Update button states
        buttons.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.mode === mode);
        });
        
        // Update dashboard layout
        dashboard.className = `dashboard mode-${mode}`;
        this.currentMode = mode;
        
        // Adjust visualization visibility based on mode
        this.adjustVisualizationVisibility();
        
        console.log(`Switched to ${mode} mode`);
    }
    
    adjustVisualizationVisibility() {
        const tier2 = document.getElementById('tier-2-visualizations');
        const expertZone = document.getElementById('expert-zone');
        
        switch(this.currentMode) {
            case 'technical':
                // Show Tier 2 visualizations immediately in technical mode
                tier2.style.maxHeight = '300px';
                tier2.style.overflow = 'visible';
                break;
            case 'streaming':
                // Hide advanced tiers, focus on hero visualizations
                tier2.style.maxHeight = '0';
                tier2.style.overflow = 'hidden';
                expertZone.classList.remove('expanded');
                break;
            case 'development':
                // Show all visualizations simultaneously
                tier2.style.maxHeight = '300px';
                tier2.style.overflow = 'visible';
                expertZone.classList.add('expanded');
                break;
        }
    }
    
    setupProgressiveDisclosure() {
        // Auto-expand Tier 2 after 30 seconds
        setTimeout(() => {
            if (this.currentMode === 'technical') {
                this.expandTier(2);
            }
        }, 30000);
    }
    
    expandTier(tier) {
        if (tier === 2) {
            const tier2 = document.getElementById('tier-2-visualizations');
            tier2.style.maxHeight = '300px';
            this.visibilityTiers.push(2);
            console.log('Tier 2 visualizations expanded');
        } else if (tier === 3) {
            const expertZone = document.getElementById('expert-zone');
            expertZone.classList.add('expanded');
            this.visibilityTiers.push(3);
            console.log('Expert zone expanded');
        }
    }
    
    setupKeyboardControls() {
        document.addEventListener('keydown', (e) => {
            switch(e.key) {
                case ' ': // Spacebar - pause animations
                    e.preventDefault();
                    this.animationCoordinator.togglePause();
                    break;
                case 'ArrowRight':
                    this.navigateVisualizationTiers(1);
                    break;
                case 'ArrowLeft':
                    this.navigateVisualizationTiers(-1);
                    break;
                case 'f':
                case 'F':
                    this.toggleFullscreen();
                    break;
                case '1': case '2': case '3': case '4': case '5': case '6': case '7': case '8':
                    this.focusVisualization(parseInt(e.key));
                    break;
            }
        });
    }
    
    setupResizeHandler() {
        let resizeTimeout;
        window.addEventListener('resize', () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                this.resizeCanvases();
            }, 250); // Debounce resize events
        });
    }
    
    resizeCanvases() {
        // Resize all initialized canvases to match their containers
        const canvasResizeMap = {
            'exploration-gauge': this.explorationGauge,
            'skill-radar': this.skillRadar,
            'gradient-flow': this.gradientFlow,
            'buffer-dynamics': this.bufferDynamics
        };
        
        Object.entries(canvasResizeMap).forEach(([canvasId, visualization]) => {
            if (visualization && visualization.canvas) {
                const canvas = visualization.canvas;
                canvas.width = canvas.parentElement.offsetWidth - 30;
                canvas.height = canvas.parentElement.offsetHeight - 30;
                
                // Re-render after resize
                if (visualization.render) {
                    visualization.render();
                }
            }
        });
    }
    
    initializeVisualizations() {
        // Tier 1: Core Learning Indicators
        this.initNeuralHeatmap();
        this.initExplorationGauge();
        this.initAdvantageOscillation();
        
        // Tier 2: Advanced Neural Visualization (initialize immediately)
        this.initTier2Visualizations();
        
        // Tier 3: Tournament Features (lazy init)
        this.setupTier3LazyInit();
    }
    
    // Tier 1 Visualizations
    initNeuralHeatmap() {
        const canvas = document.getElementById('neural-heatmap');
        if (!canvas) return;
        
        canvas.width = 360;
        canvas.height = 360;
        const ctx = canvas.getContext('2d');
        
        this.neuralHeatmap = {
            canvas: canvas,
            ctx: ctx,
            confidenceData: new Array(81).fill(0.5), // 9x9 board
            render: () => this.renderNeuralHeatmap()
        };
        
        // Initial render to show the heatmap overlay immediately
        this.neuralHeatmap.render();
        
        console.log('Neural heatmap initialized and rendered');
    }
    
    renderNeuralHeatmap() {
        if (!this.neuralHeatmap) return;
        
        const { ctx, confidenceData } = this.neuralHeatmap;
        const cellSize = 40;
        
        ctx.clearRect(0, 0, 360, 360);
        
        // Render confidence overlay
        for (let i = 0; i < 9; i++) {
            for (let j = 0; j < 9; j++) {
                const index = i * 9 + j;
                const confidence = confidenceData[index];
                
                // Color mapping: red (low confidence) to blue (high confidence)
                const red = Math.floor(255 * (1 - confidence));
                const blue = Math.floor(255 * confidence);
                const alpha = 0.4; // Semi-transparent
                
                ctx.fillStyle = `rgba(${red}, 0, ${blue}, ${alpha})`;
                ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
            }
        }
    }
    
    initExplorationGauge() {
        const canvas = document.getElementById('exploration-gauge');
        if (!canvas) return;
        
        // Make canvas responsive to container
        canvas.width = canvas.parentElement.offsetWidth - 30;
        canvas.height = canvas.parentElement.offsetHeight - 30;
        
        const ctx = canvas.getContext('2d');
        
        this.explorationGauge = {
            canvas: canvas,
            ctx: ctx,
            entropy: 0.5, // Current entropy value
            render: () => this.renderExplorationGauge()
        };
        
        // Initial render to show the gauge immediately
        this.explorationGauge.render();
        
        console.log('Exploration gauge initialized and rendered');
    }
    
    renderExplorationGauge() {
        if (!this.explorationGauge) return;
        
        const { ctx, entropy } = this.explorationGauge;
        const centerX = 75;
        const centerY = 60;
        const radius = 50;
        
        ctx.clearRect(0, 0, 150, 80);
        
        // Draw gauge background
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, Math.PI, 2 * Math.PI);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.lineWidth = 8;
        ctx.stroke();
        
        // Draw entropy arc
        const angle = Math.PI + (entropy * Math.PI);
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, Math.PI, angle);
        
        // Color transition: red (explore) to blue (exploit)
        const red = Math.floor(255 * entropy);
        const blue = Math.floor(255 * (1 - entropy));
        ctx.strokeStyle = `rgb(${red}, 100, ${blue})`;
        ctx.lineWidth = 8;
        ctx.stroke();
        
        // Draw needle
        const needleAngle = Math.PI + (entropy * Math.PI);
        const needleX = centerX + Math.cos(needleAngle) * (radius - 10);
        const needleY = centerY + Math.sin(needleAngle) * (radius - 10);
        
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.lineTo(needleX, needleY);
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw value text
        ctx.fillStyle = '#ffffff';
        ctx.font = '12px Monaco';
        ctx.textAlign = 'center';
        ctx.fillText(`${entropy.toFixed(3)}`, centerX, centerY + 25);
    }
    
    initAdvantageOscillation() {
        const canvas = document.getElementById('advantage-chart');
        if (!canvas) return;
        
        canvas.width = canvas.parentElement.offsetWidth;
        canvas.height = 60;
        const ctx = canvas.getContext('2d');
        
        this.advantageChart = {
            canvas: canvas,
            ctx: ctx,
            advantageHistory: [],
            render: () => this.renderAdvantageOscillation()
        };
        
        console.log('Advantage oscillation initialized');
    }
    
    renderAdvantageOscillation() {
        if (!this.advantageChart) return;
        
        const { ctx, advantageHistory } = this.advantageChart;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const centerY = height / 2;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw center line
        ctx.beginPath();
        ctx.moveTo(0, centerY);
        ctx.lineTo(width, centerY);
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();
        
        if (advantageHistory.length < 2) return;
        
        // Draw advantage oscillation
        ctx.beginPath();
        const stepSize = width / Math.max(advantageHistory.length - 1, 1);
        
        advantageHistory.forEach((advantage, i) => {
            const x = i * stepSize;
            const y = centerY - (advantage * centerY); // Scale to canvas height
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        
        // Color based on current advantage
        const currentAdvantage = advantageHistory[advantageHistory.length - 1] || 0;
        if (currentAdvantage > 0) {
            ctx.strokeStyle = '#4caf50'; // Green for black advantage
        } else {
            ctx.strokeStyle = '#f44336'; // Red for white advantage
        }
        ctx.lineWidth = 2;
        ctx.stroke();
    }
    
    // Tier 2 lazy initialization removed - now initialized immediately in initializeVisualizations()
    
    initTier2Visualizations() {
        this.initSkillRadar();
        this.initGradientFlow();
        this.initBufferDynamics();
        
        // Initial render with default data to show the charts immediately
        if (this.skillRadar) this.skillRadar.render();
        if (this.gradientFlow) this.gradientFlow.render();
        if (this.bufferDynamics) this.bufferDynamics.render();
        
        console.log('Tier 2 visualizations initialized and rendered');
    }
    
    initSkillRadar() {
        const canvas = document.getElementById('skill-radar');
        if (!canvas) return;
        
        // Make canvas responsive to container
        canvas.width = canvas.parentElement.offsetWidth - 30;
        canvas.height = canvas.parentElement.offsetHeight - 30;
        
        const ctx = canvas.getContext('2d');
        
        this.skillRadar = {
            canvas: canvas,
            ctx: ctx,
            skills: {
                opening: 0.5,
                tactics: 0.5, 
                endgame: 0.5,
                strategy: 0.5,
                pattern: 0.5,
                time: 0.5
            },
            render: () => this.renderSkillRadar()
        };
    }
    
    renderSkillRadar() {
        if (!this.skillRadar) return;
        
        const { ctx, skills } = this.skillRadar;
        const centerX = 100;
        const centerY = 100;
        const radius = 80;
        
        ctx.clearRect(0, 0, 200, 200);
        
        const skillNames = Object.keys(skills);
        const angleStep = (2 * Math.PI) / skillNames.length;
        
        // Draw radar grid
        for (let i = 1; i <= 5; i++) {
            ctx.beginPath();
            const r = (radius / 5) * i;
            
            skillNames.forEach((_, index) => {
                const angle = index * angleStep - Math.PI / 2;
                const x = centerX + Math.cos(angle) * r;
                const y = centerY + Math.sin(angle) * r;
                
                if (index === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            
            ctx.closePath();
            ctx.strokeStyle = `rgba(255, 255, 255, ${0.1 + i * 0.05})`;
            ctx.lineWidth = 1;
            ctx.stroke();
        }
        
        // Draw skill polygon
        ctx.beginPath();
        skillNames.forEach((skill, index) => {
            const angle = index * angleStep - Math.PI / 2;
            const skillValue = skills[skill];
            const x = centerX + Math.cos(angle) * (radius * skillValue);
            const y = centerY + Math.sin(angle) * (radius * skillValue);
            
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        
        ctx.closePath();
        ctx.fillStyle = 'rgba(76, 205, 196, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#4ecdc4';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw skill labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '10px Monaco';
        ctx.textAlign = 'center';
        
        skillNames.forEach((skill, index) => {
            const angle = index * angleStep - Math.PI / 2;
            const x = centerX + Math.cos(angle) * (radius + 20);
            const y = centerY + Math.sin(angle) * (radius + 20);
            
            ctx.fillText(skill.charAt(0).toUpperCase() + skill.slice(1), x, y);
        });
    }
    
    initGradientFlow() {
        const canvas = document.getElementById('gradient-flow');
        if (!canvas) return;
        
        canvas.width = canvas.parentElement.offsetWidth - 30;
        canvas.height = canvas.parentElement.offsetHeight - 30;
        
        this.gradientFlow = {
            canvas: canvas,
            ctx: canvas.getContext('2d'),
            layers: ['input', 'conv1', 'conv2', 'fc1', 'fc2', 'output'],
            gradients: [0.1, 0.2, 0.15, 0.3, 0.25, 0.1],
            particles: [],
            render: () => this.renderGradientFlow()
        };
        
        // Initialize particles
        for (let i = 0; i < 20; i++) {
            this.gradientFlow.particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 2,
                vy: (Math.random() - 0.5) * 2,
                life: Math.random()
            });
        }
    }
    
    renderGradientFlow() {
        if (!this.gradientFlow) return;
        
        const { ctx, layers, gradients, particles } = this.gradientFlow;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw layer connections
        const layerWidth = width / layers.length;
        
        for (let i = 0; i < layers.length - 1; i++) {
            const x1 = (i + 0.5) * layerWidth;
            const x2 = (i + 1.5) * layerWidth;
            const gradient = gradients[i];
            
            ctx.beginPath();
            ctx.moveTo(x1, height / 2);
            ctx.lineTo(x2, height / 2);
            ctx.strokeStyle = `rgba(255, ${Math.floor(255 * gradient)}, 100, 0.8)`;
            ctx.lineWidth = Math.max(1, gradient * 10);
            ctx.stroke();
        }
        
        // Update and draw particles
        particles.forEach(particle => {
            particle.x += particle.vx;
            particle.y += particle.vy;
            particle.life -= 0.01;
            
            if (particle.life <= 0 || particle.x < 0 || particle.x > width || particle.y < 0 || particle.y > height) {
                particle.x = Math.random() * width;
                particle.y = Math.random() * height;
                particle.life = 1;
            }
            
            ctx.beginPath();
            ctx.arc(particle.x, particle.y, 2, 0, 2 * Math.PI);
            ctx.fillStyle = `rgba(76, 205, 196, ${particle.life})`;
            ctx.fill();
        });
    }
    
    initBufferDynamics() {
        const canvas = document.getElementById('buffer-dynamics');
        if (!canvas) return;
        
        canvas.width = canvas.parentElement.offsetWidth - 30;
        canvas.height = canvas.parentElement.offsetHeight - 30;
        
        this.bufferDynamics = {
            canvas: canvas,
            ctx: canvas.getContext('2d'),
            bufferData: {
                size: 0,
                capacity: 1000,
                qualityDistribution: new Array(10).fill(0)
            },
            render: () => this.renderBufferDynamics()
        };
    }
    
    renderBufferDynamics() {
        if (!this.bufferDynamics) return;
        
        const { ctx, bufferData } = this.bufferDynamics;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw buffer utilization
        const utilizationWidth = (bufferData.size / bufferData.capacity) * width;
        
        ctx.fillStyle = 'rgba(76, 205, 196, 0.3)';
        ctx.fillRect(0, 0, utilizationWidth, height / 2);
        
        ctx.strokeStyle = '#4ecdc4';
        ctx.strokeRect(0, 0, width, height / 2);
        
        // Draw quality distribution histogram
        const barWidth = width / bufferData.qualityDistribution.length;
        const maxQuality = Math.max(...bufferData.qualityDistribution, 1);
        
        bufferData.qualityDistribution.forEach((quality, i) => {
            const barHeight = (quality / maxQuality) * (height / 2);
            const x = i * barWidth;
            const y = height - barHeight;
            
            ctx.fillStyle = `rgba(255, ${Math.floor(255 * (i / 10))}, 100, 0.7)`;
            ctx.fillRect(x, y, barWidth - 1, barHeight);
        });
    }
    
    setupTier3LazyInit() {
        // Initialize Tier 3 visualizations when expert zone is expanded
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !this.tournamentTree) {
                    this.initTier3Visualizations();
                    observer.disconnect();
                }
            });
        });
        
        const expertZone = document.getElementById('expert-zone');
        if (expertZone) {
            observer.observe(expertZone);
        }
    }
    
    initTier3Visualizations() {
        this.initTournamentTree();
        this.initStrategicFingerprinting();
        console.log('Tier 3 visualizations initialized');
    }
    
    initTournamentTree() {
        const canvas = document.getElementById('tournament-tree');
        if (!canvas) return;
        
        canvas.width = canvas.parentElement.offsetWidth - 30;
        canvas.height = 200;
        
        this.tournamentTree = {
            canvas: canvas,
            ctx: canvas.getContext('2d'),
            tournaments: [
                { name: 'Checkpoint A', elo: 1200, x: 50, y: 50 },
                { name: 'Checkpoint B', elo: 1350, x: 150, y: 50 },
                { name: 'Checkpoint C', elo: 1480, x: 250, y: 50 },
                { name: 'Current', elo: 1520, x: 350, y: 50 }
            ],
            connections: [[0, 1], [1, 2], [2, 3]],
            render: () => this.renderTournamentTree()
        };
    }
    
    renderTournamentTree() {
        if (!this.tournamentTree) return;
        
        const { ctx, tournaments, connections } = this.tournamentTree;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw connections
        connections.forEach(([from, to]) => {
            const fromNode = tournaments[from];
            const toNode = tournaments[to];
            
            ctx.beginPath();
            ctx.moveTo(fromNode.x, fromNode.y + 30);
            ctx.lineTo(toNode.x, toNode.y + 30);
            
            // Color based on ELO progression
            const eloGain = toNode.elo - fromNode.elo;
            if (eloGain > 0) {
                ctx.strokeStyle = '#4caf50'; // Green for improvement
            } else {
                ctx.strokeStyle = '#f44336'; // Red for decline
            }
            ctx.lineWidth = Math.abs(eloGain) / 50; // Width based on change
            ctx.stroke();
        });
        
        // Draw tournament nodes
        tournaments.forEach((tournament, i) => {
            const { x, y, name, elo } = tournament;
            
            // Draw node circle
            ctx.beginPath();
            ctx.arc(x, y + 30, 20, 0, 2 * Math.PI);
            ctx.fillStyle = i === tournaments.length - 1 ? '#4ecdc4' : 'rgba(255, 255, 255, 0.8)';
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw labels
            ctx.fillStyle = '#ffffff';
            ctx.font = '10px Monaco';
            ctx.textAlign = 'center';
            ctx.fillText(name, x, y + 15);
            ctx.fillText(`${elo}`, x, y + 70);
        });
    }
    
    initStrategicFingerprinting() {
        const canvas = document.getElementById('strategy-radar');
        if (!canvas) return;
        
        canvas.width = canvas.parentElement.offsetWidth - 30;
        canvas.height = 200;
        
        this.strategyRadar = {
            canvas: canvas,
            ctx: canvas.getContext('2d'),
            strategies: {
                aggressive: 0.6,
                defensive: 0.4,
                positional: 0.7,
                tactical: 0.8,
                material: 0.5,
                initiative: 0.6
            },
            render: () => this.renderStrategicFingerprinting()
        };
    }
    
    renderStrategicFingerprinting() {
        if (!this.strategyRadar) return;
        
        const { ctx, strategies } = this.strategyRadar;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const radius = 80;
        
        ctx.clearRect(0, 0, width, height);
        
        const strategyNames = Object.keys(strategies);
        const angleStep = (2 * Math.PI) / strategyNames.length;
        
        // Draw strategic fingerprint polygon
        ctx.beginPath();
        strategyNames.forEach((strategy, index) => {
            const angle = index * angleStep - Math.PI / 2;
            const value = strategies[strategy];
            const x = centerX + Math.cos(angle) * (radius * value);
            const y = centerY + Math.sin(angle) * (radius * value);
            
            if (index === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        });
        
        ctx.closePath();
        ctx.fillStyle = 'rgba(255, 107, 107, 0.3)';
        ctx.fill();
        ctx.strokeStyle = '#ff6b6b';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw strategy labels
        ctx.fillStyle = '#ffffff';
        ctx.font = '9px Monaco';
        ctx.textAlign = 'center';
        
        strategyNames.forEach((strategy, index) => {
            const angle = index * angleStep - Math.PI / 2;
            const x = centerX + Math.cos(angle) * (radius + 25);
            const y = centerY + Math.sin(angle) * (radius + 25);
            
            ctx.fillText(strategy.charAt(0).toUpperCase() + strategy.slice(1), x, y);
        });
    }
    
    // Data update methods
    updateNeuralConfidence(confidenceData) {
        if (this.neuralHeatmap && confidenceData) {
            this.neuralHeatmap.confidenceData = confidenceData;
            this.animationCoordinator.scheduleUpdate('neural-heatmap', this.neuralHeatmap.render);
        }
    }
    
    updateExploration(entropy) {
        if (this.explorationGauge && typeof entropy === 'number') {
            this.explorationGauge.entropy = entropy;
            this.dataBuffers.exploration.push(entropy);
            if (this.dataBuffers.exploration.length > 100) {
                this.dataBuffers.exploration.shift();
            }
            this.animationCoordinator.scheduleUpdate('exploration-gauge', this.explorationGauge.render);
        }
    }
    
    updateAdvantage(advantage) {
        if (this.advantageChart && typeof advantage === 'number') {
            this.dataBuffers.advantage.push(advantage);
            if (this.dataBuffers.advantage.length > 200) {
                this.dataBuffers.advantage.shift();
            }
            this.advantageChart.advantageHistory = this.dataBuffers.advantage;
            this.animationCoordinator.scheduleUpdate('advantage-chart', this.advantageChart.render);
        }
    }
    
    updateSkills(skillData) {
        if (this.skillRadar && skillData) {
            Object.assign(this.skillRadar.skills, skillData);
            this.animationCoordinator.scheduleUpdate('skill-radar', this.skillRadar.render);
        }
    }
    
    updateGradients(gradientData) {
        if (this.gradientFlow && gradientData) {
            this.gradientFlow.gradients = gradientData;
            this.animationCoordinator.scheduleUpdate('gradient-flow', this.gradientFlow.render);
        }
    }
    
    updateBuffer(bufferInfo) {
        if (this.bufferDynamics && bufferInfo) {
            Object.assign(this.bufferDynamics.bufferData, bufferInfo);
            this.animationCoordinator.scheduleUpdate('buffer-dynamics', this.bufferDynamics.render);
        }
    }
    
    // Navigation and interaction methods
    navigateVisualizationTiers(direction) {
        // Implementation for arrow key navigation
        console.log(`Navigating visualization tiers: ${direction}`);
    }
    
    focusVisualization(number) {
        // Implementation for number key shortcuts
        console.log(`Focusing visualization: ${number}`);
    }
    
    toggleFullscreen() {
        if (document.fullscreenElement) {
            document.exitFullscreen();
        } else {
            document.documentElement.requestFullscreen();
        }
    }
}

// Animation Coordinator - manages smooth updates and prevents conflicts
class AnimationCoordinator {
    constructor() {
        this.updateQueue = new Map();
        this.isPaused = false;
        this.lastUpdate = 0;
        this.targetFPS = 30;
        this.frameInterval = 1000 / this.targetFPS;
        
        this.startAnimationLoop();
    }
    
    scheduleUpdate(id, renderFunction) {
        if (this.isPaused) return;
        
        this.updateQueue.set(id, {
            render: renderFunction,
            priority: this.getPriority(id),
            timestamp: Date.now()
        });
    }
    
    getPriority(id) {
        // Tier 1 visualizations get higher priority
        const tier1 = ['neural-heatmap', 'exploration-gauge', 'advantage-chart'];
        return tier1.includes(id) ? 1 : 2;
    }
    
    startAnimationLoop() {
        const animate = (currentTime) => {
            if (currentTime - this.lastUpdate >= this.frameInterval) {
                this.processUpdateQueue();
                this.lastUpdate = currentTime;
            }
            requestAnimationFrame(animate);
        };
        
        requestAnimationFrame(animate);
    }
    
    processUpdateQueue() {
        if (this.isPaused || this.updateQueue.size === 0) return;
        
        // Sort by priority and process
        const sortedUpdates = Array.from(this.updateQueue.entries())
            .sort(([, a], [, b]) => a.priority - b.priority);
        
        // Limit to 3 updates per frame to maintain performance
        sortedUpdates.slice(0, 3).forEach(([id, update]) => {
            try {
                update.render();
            } catch (error) {
                console.error(`Error rendering ${id}:`, error);
            }
        });
        
        this.updateQueue.clear();
    }
    
    togglePause() {
        this.isPaused = !this.isPaused;
        console.log(`Animation ${this.isPaused ? 'paused' : 'resumed'}`);
    }
}

// Performance monitoring is handled by the main WebUI app.js PerformanceMonitor class
// to avoid class name conflicts and duplication

// Initialize the advanced visualization system when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    try {
        console.log('🚀 Initializing AdvancedVisualizationManager...');
        window.advancedVisualizations = new AdvancedVisualizationManager();
        console.log('✅ AdvancedVisualizationManager initialized successfully');
    } catch (error) {
        console.error('❌ Failed to initialize AdvancedVisualizationManager:', error);
        console.error('Stack:', error.stack);
    }
});