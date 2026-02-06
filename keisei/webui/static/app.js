// Keisei WebUI JavaScript Application
class KeiseiWebUI {
    constructor() {
        this.websocket = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 2000;
        
        // Chart instances
        this.learningChart = null;
        this.winrateChart = null;
        this.ppoChart = null;
        this.episodeChart = null;
        
        // Historical data accumulation for trend charts
        this.historicalData = {
            timestamps: [],
            metrics: new Map() // metric_name -> array of values
        };
        this.maxHistoryLength = 360; // Keep 6 hours of data (1 point per minute = 360 points)
        this.miniCharts = new Map(); // Store mini chart instances to prevent recreation
        
        // Memory management and performance monitoring (initialized later)
        this.memoryManager = null;
        this.performanceMonitor = null;
        this.updateThrottler = null;
        this.isLowPerformance = false;
        
        // Rate limiting for updates
        this.lastUIUpdate = 0;
        this.uiUpdateInterval = 100; // Minimum 100ms between UI updates
        
        // Historical data throttling for 6-hour trends
        this.lastHistoryUpdate = 0;
        this.historyUpdateInterval = 60000; // Store data points every 60 seconds (1 minute) for meaningful trends
        
        // Cleanup tracking
        this.activeElements = new Set();
        this.cleanupTasks = [];
        
        // Feature flags for performance degradation
        this.featuresEnabled = {
            miniCharts: true,  // Re-enabled - these are the metric trend charts
            animations: true,
            advancedVisualizations: true,
            excessiveLogging: false  // Disable by default
        };
        
        // Unicode pieces mapping (fixed to match backend underscore format)
        this.pieces = {
            'pawn': { black: '歩', white: '歩' },
            'lance': { black: '香', white: '香' },
            'knight': { black: '桂', white: '桂' },
            'silver': { black: '銀', white: '銀' },
            'gold': { black: '金', white: '金' },
            'bishop': { black: '角', white: '角' },
            'rook': { black: '飛', white: '飛' },
            'king': { black: '王', white: '王' },
            'promoted_pawn': { black: 'と', white: 'と' },
            'promoted_lance': { black: '成香', white: '成香' },
            'promoted_knight': { black: '成桂', white: '成桂' },
            'promoted_silver': { black: '成銀', white: '成銀' },
            'promoted_bishop': { black: '馬', white: '馬' },
            'promoted_rook': { black: '龍', white: '龍' }
        };

        this.init();
    }

    init() {
        this.createBoard();
        this.setupCharts();
        this.connect();
    }

    connect() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.hostname;
        const port = 8765; // WebSocket port
        const url = `${protocol}//${host}:${port}`;

        console.log(`Connecting to ${url}`);
        
        try {
            this.websocket = new WebSocket(url);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                
                // Initialize performance monitoring systems after connection
                if (!this.memoryManager) {
                    this.memoryManager = new MemoryManager(this);
                    this.performanceMonitor = new PerformanceMonitor(this);
                    this.updateThrottler = new UpdateThrottler();
                    console.log('✅ Performance monitoring initialized');
                }
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const message = JSON.parse(event.data);
                    this.handleMessage(message);
                } catch (error) {
                    console.error('Error parsing message:', error);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateConnectionStatus(false);
                this.attemptReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to create WebSocket:', error);
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        }
    }

    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connect(), this.reconnectDelay);
        }
    }

    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        statusElement.textContent = connected ? 'Connected' : 'Disconnected';
        statusElement.className = `connection-status ${connected ? 'connected' : 'disconnected'}`;
    }

    createBoard() {
        const board = document.getElementById('shogi-board');
        board.innerHTML = '';
        
        // Create 9x9 grid of cells
        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9; col++) {
                const cell = document.createElement('div');
                cell.className = 'board-cell';
                cell.id = `cell-${row}-${col}`;
                board.appendChild(cell);
            }
        }
    }

    setupCharts() {
        const commonOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            scales: {
                x: { display: false },
                y: {
                    beginAtZero: false,
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    ticks: { color: '#ffffff', font: { size: 10 } }
                }
            },
            plugins: {
                title: {
                    display: true,
                    color: '#ffffff',
                    font: { size: 12, weight: 'bold' },
                    padding: { top: 5, bottom: 10 }
                },
                legend: {
                    display: true,
                    labels: { color: '#ffffff', font: { size: 10 } }
                }
            }
        };

        // 1. Learning Progress Chart (Loss + Entropy)
        this.learningChart = new Chart(document.getElementById('learning-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Policy Loss',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'Value Loss',
                        data: [],
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(76, 205, 196, 0.1)',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'Entropy',
                        data: [],
                        borderColor: '#45b7d1',
                        backgroundColor: 'rgba(69, 183, 209, 0.1)',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                ...commonOptions,
                plugins: {
                    ...commonOptions.plugins,
                    title: {
                        ...commonOptions.plugins.title,
                        text: 'Policy Loss, Value Loss & Entropy'
                    }
                }
            }
        });

        // 2. Win Rate Chart
        this.winrateChart = new Chart(document.getElementById('winrate-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Win %',
                        data: [],
                        borderColor: '#4ecdc4',
                        backgroundColor: 'rgba(76, 205, 196, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        borderWidth: 2
                    },
                    {
                        label: 'Loss %', 
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.1,
                        pointRadius: 2,
                        borderWidth: 2
                    },
                    {
                        label: 'Draw %',
                        data: [],
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)', 
                        tension: 0.1,
                        pointRadius: 2,
                        borderWidth: 2
                    }
                ]
            },
            options: { 
                ...commonOptions, 
                scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, beginAtZero: true, max: 100 } },
                plugins: {
                    ...commonOptions.plugins,
                    title: {
                        ...commonOptions.plugins.title,
                        text: 'Win Rate Trends Over Time'
                    }
                }
            }
        });

        // 3. PPO Training Chart
        this.ppoChart = new Chart(document.getElementById('ppo-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'KL Divergence',
                        data: [],
                        borderColor: '#9c27b0',
                        backgroundColor: 'rgba(156, 39, 176, 0.1)',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'Clip Fraction',
                        data: [],
                        borderColor: '#ff9800',
                        backgroundColor: 'rgba(255, 152, 0, 0.1)',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                ...commonOptions,
                plugins: {
                    ...commonOptions.plugins,
                    title: {
                        ...commonOptions.plugins.title,
                        text: 'PPO Training Metrics'
                    }
                }
            }
        });

        // 4. Episode Performance Chart
        this.episodeChart = new Chart(document.getElementById('episode-chart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Episode Length',
                        data: [],
                        borderColor: '#00bcd4',
                        backgroundColor: 'rgba(0, 188, 212, 0.1)',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2
                    },
                    {
                        label: 'Episode Reward',
                        data: [],
                        borderColor: '#8bc34a',
                        backgroundColor: 'rgba(139, 195, 74, 0.1)',
                        tension: 0.1,
                        pointRadius: 0,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                ...commonOptions,
                plugins: {
                    ...commonOptions.plugins,
                    title: {
                        ...commonOptions.plugins.title,
                        text: 'Episode Length & Rewards'
                    }
                }
            }
        });
    }

    handleMessage(message) {
        // Rate-limited logging to prevent memory pressure
        if (this.featuresEnabled.excessiveLogging) {
            console.log('📨 Received message:', message.type);
        }
        
        // Throttle UI updates to prevent overload
        const now = Date.now();
        if (now - this.lastUIUpdate < this.uiUpdateInterval && !this.isLowPerformance) {
            if (this.updateThrottler) {
                this.updateThrottler.scheduleUpdate(() => this._processMessage(message));
                return;
            }
        }
        
        this.lastUIUpdate = now;
        this._processMessage(message);
    }
    
    _processMessage(message) {
        try {
            // Track memory usage periodically
            if (Math.random() < 0.01 && this.memoryManager) { // 1% of messages
                this.memoryManager.checkMemoryUsage();
            }
            
            switch (message.type) {
                case 'connected':
                    if (this.featuresEnabled.excessiveLogging) {
                        console.log('Connection confirmed:', message.message);
                    }
                    break;
                    
                case 'progress_update':
                    this.updateProgress(message.data);
                    if (this.featuresEnabled.advancedVisualizations) {
                        this.updateAdvancedVisualizations(message.data);
                    }
                    break;
                    
                case 'board_update':
                    this.updateBoard(message.data);
                    break;
                    
                case 'metrics_update':
                    this.updateMetrics(message.data);
                    if (this.featuresEnabled.advancedVisualizations) {
                        this.updateAdvancedVisualizations(message.data);
                    }
                    break;
                    
                default:
                    if (this.featuresEnabled.excessiveLogging) {
                        console.log('Unknown message type:', message.type);
                    }
            }
        } catch (error) {
            console.error('❌ Error handling message:', error);
            this.showErrorIndicator('Message processing error');
        }
    }

    updateProgress(data) {
        try {
            // Update header stats
            this.safeUpdateElement('timestep', this.formatNumber(data.global_timestep || 0));
            this.safeUpdateElement('episodes', this.formatNumber(data.total_episodes || 0));
            this.safeUpdateElement('speed', (data.speed || 0).toFixed(1));
            
            // Update win/loss stats
            this.safeUpdateElement('black-wins', data.black_wins || 0);
            this.safeUpdateElement('white-wins', data.white_wins || 0);
            this.safeUpdateElement('draws', data.draws || 0);
            
            // Calculate and update win rate
            const totalGames = (data.black_wins || 0) + (data.white_wins || 0) + (data.draws || 0);
            const blackWinRate = totalGames > 0 ? ((data.black_wins || 0) / totalGames * 100).toFixed(1) : 0;
            this.safeUpdateElement('win-rate', `${blackWinRate}%`);
            
            // CRITICAL FIX: Update detailed training metrics with actual data
            // Extract meaningful episode and PPO info from available data
            const episodeInfo = `${data.total_episodes || 0} episodes (${data.speed?.toFixed(1) || 0.0} steps/sec)`;
            const ppoInfo = `Training active (Step ${data.global_timestep || 0})`;
            
            this.safeUpdateElement('ep-metrics', episodeInfo);
            this.safeUpdateElement('ppo-metrics', ppoInfo);
            this.safeUpdateElement('current-epoch', data.current_epoch || 0);
        } catch (error) {
            console.error('❌ Error in updateProgress:', error);
            this.showErrorIndicator('Progress update error');
        }
    }

    updateBoard(data) {
        try {
            if (!data.board) return;
            
            // Clear all cells
            const cells = document.querySelectorAll('.board-cell');
            cells.forEach(cell => {
                cell.innerHTML = '';
                cell.className = 'board-cell';
            });
        
        // Update board pieces
        for (let row = 0; row < 9; row++) {
            for (let col = 0; col < 9; col++) {
                const cell = document.getElementById(`cell-${row}-${col}`);
                const piece = data.board[row][col];
                
                if (piece) {
                    const pieceImg = document.createElement('img');
                    
                    // Use SVG images instead of text
                    const imageName = `${piece.type}_${piece.color}.svg`;
                    pieceImg.src = `images/${imageName}`;
                    pieceImg.alt = `${piece.type} ${piece.color}`;
                    pieceImg.className = 'piece-image';
                    
                    // Rotate white pieces (SVG handles the visual correctly)
                    if (piece.color === 'white') {
                        pieceImg.style.transform = 'rotate(180deg)';
                    }
                    
                    // Handle missing images gracefully
                    pieceImg.onerror = () => {
                        // Fallback to text if image fails to load
                        const textElement = document.createElement('div');
                        textElement.className = `piece ${piece.color}`;
                        textElement.textContent = this.pieces[piece.type]?.[piece.color] || piece.type;
                        cell.replaceChild(textElement, pieceImg);
                    };
                    
                    cell.appendChild(pieceImg);
                }
            }
        }
        
        // Highlight hot squares
        if (data.hot_squares) {
            data.hot_squares.forEach(squareName => {
                const coords = this.squareNameToCoords(squareName);
                if (coords) {
                    const cell = document.getElementById(`cell-${coords.row}-${coords.col}`);
                    if (cell) {
                        cell.classList.add('hot-square');
                    }
                }
            });
        }
        
        // Update game status
        const statusElement = document.getElementById('game-status');
        if (data.game_over) {
            statusElement.textContent = data.winner ? 
                `Game Over - ${data.winner.charAt(0).toUpperCase() + data.winner.slice(1)} Wins!` :
                'Game Over - Draw';
            statusElement.className = 'game-status finished';
        } else {
            const currentPlayer = data.current_player?.charAt(0).toUpperCase() + data.current_player?.slice(1) || 'Unknown';
            statusElement.textContent = `${currentPlayer} to move (Move ${data.move_count || 0})`;
            statusElement.className = 'game-status playing';
        }
        
        // Update hands (captured pieces)
        this.updateHand('sente-hand', data.sente_hand || {});
        this.updateHand('gote-hand', data.gote_hand || {});
        
        // Update recent moves
        this.updateRecentMoves(data.recent_moves || []);
        
        // Update hot squares display
        this.updateHotSquares(data.hot_squares || []);
        } catch (error) {
            console.error('❌ Error in updateBoard:', error);
            this.showErrorIndicator('Board update error');
        }
    }

    updateMetrics(data) {
        try {
            if (!data.learning_curves) {
                console.log('No learning_curves data available');
                return;
            }
            
            const curves = data.learning_curves;
            const maxLength = Math.max(
                curves.policy_losses?.length || 0,
                curves.value_losses?.length || 0,
                curves.entropies?.length || 0,
                curves.kl_divergences?.length || 0,
                curves.clip_fractions?.length || 0,
                curves.episode_lengths?.length || 0,
                curves.episode_rewards?.length || 0
            );
        
            if (maxLength > 0) {
                const labels = Array.from({length: maxLength}, (_, i) => i);
                
                // 1. Update Learning Progress Chart
                try {
                    if (this.learningChart) {
                        this.learningChart.data.labels = labels;
                        this.learningChart.data.datasets[0].data = curves.policy_losses || [];
                        this.learningChart.data.datasets[1].data = curves.value_losses || [];
                        this.learningChart.data.datasets[2].data = curves.entropies || [];
                        this.learningChart.update('none');
                    }
                } catch (chartError) {
                    console.warn('Learning chart update failed:', chartError);
                }
                
                // 2. Update Win Rate Chart
                try {
                    if (this.winrateChart && data.game_statistics?.win_loss_draw_rates) {
                        const rates = data.game_statistics.win_loss_draw_rates;
                        
                        // Backend sends percentages (0.0-1.0), convert to percentage values (0-100)
                        const winRate = (rates.win || 0) * 100;
                        const lossRate = (rates.loss || 0) * 100;  
                        const drawRate = (rates.draw || 0) * 100;
                        
                        // Accumulate historical win rate data for trending
                        this.accumulateWinRateHistory(winRate, lossRate, drawRate);
                        
                        // Get historical data for chart
                        const winHistory = this.historicalData.metrics.get('win_rate') || [];
                        const lossHistory = this.historicalData.metrics.get('loss_rate') || [];
                        const drawHistory = this.historicalData.metrics.get('draw_rate') || [];
                        
                        if (winHistory.length > 0) {
                            // Create labels based on historical data length
                            const labels = Array.from({length: winHistory.length}, (_, i) => i);
                            
                            // Update chart with historical trends
                            this.winrateChart.data.labels = labels;
                            this.winrateChart.data.datasets[0].data = winHistory; // Win %
                            this.winrateChart.data.datasets[1].data = lossHistory; // Loss %
                            this.winrateChart.data.datasets[2].data = drawHistory; // Draw %
                            this.winrateChart.update('none');
                        }
                    }
                } catch (chartError) {
                    console.warn('Win rate chart update failed:', chartError);
                }
                
                // 3. Update PPO Training Chart
                try {
                    if (this.ppoChart) {
                        this.ppoChart.data.labels = labels;
                        this.ppoChart.data.datasets[0].data = curves.kl_divergences || [];
                        this.ppoChart.data.datasets[1].data = curves.clip_fractions || [];
                        this.ppoChart.update('none');
                    }
                } catch (chartError) {
                    console.warn('PPO chart update failed:', chartError);
                }
                
                // 4. Update Episode Performance Chart
                try {
                    if (this.episodeChart) {
                        this.episodeChart.data.labels = labels;
                        this.episodeChart.data.datasets[0].data = curves.episode_lengths || [];
                        this.episodeChart.data.datasets[1].data = curves.episode_rewards || [];
                        this.episodeChart.update('none');
                    }
                } catch (chartError) {
                    console.warn('Episode chart update failed:', chartError);
                }
            }
        
        // Update text statistics
        if (data.game_statistics) {
            const stats = data.game_statistics;
            document.getElementById('games-per-hour').textContent = (stats.games_per_hour || 0).toFixed(1);
            document.getElementById('avg-game-length').textContent = Math.round(stats.average_game_length || 0);
        }
        
        // Update buffer progress
        if (data.buffer_info) {
            const capacity = data.buffer_info.buffer_capacity || 1;
            const size = data.buffer_info.buffer_size || 0;
            const percentage = Math.min((size / capacity) * 100, 100);
            
            const progressBar = document.getElementById('buffer-progress');
            const progressText = document.getElementById('buffer-text');
            if (progressBar) {
                progressBar.style.width = `${percentage}%`;
            }
            if (progressText) {
                progressText.textContent = `${size} / ${capacity}`;
            }
        }
        
        // Update gradient norm (was missing!)
        if (data.model_info && data.model_info.gradient_norm !== undefined) {
            const gradientElement = document.getElementById('gradient-norm');
            if (gradientElement) {
                gradientElement.textContent = data.model_info.gradient_norm.toFixed(4);
            }
        }
        
        // Render metrics table (was completely missing!)
        if (data.metrics_table) {
            this.renderMetricsTable(data.metrics_table);
        }
        
        // Show/hide processing indicator
        const processingIndicator = document.getElementById('processing-indicator');
        if (data.processing) {
            processingIndicator.style.display = 'block';
        } else {
            processingIndicator.style.display = 'none';
        }
        } catch (error) {
            console.error('❌ Error in updateMetrics:', error);
            this.showErrorIndicator('Metrics update error');
        }
    }

    updateHand(elementId, hand) {
        const element = document.getElementById(elementId);
        element.innerHTML = '';
        
        if (Object.keys(hand).length === 0) {
            element.innerHTML = '<span style="opacity: 0.5; font-size: 12px;">None</span>';
            return;
        }
        
        Object.entries(hand).forEach(([pieceType, count]) => {
            const pieceContainer = document.createElement('span');
            pieceContainer.style.display = 'inline-flex';
            pieceContainer.style.alignItems = 'center';
            pieceContainer.style.marginRight = '8px';
            pieceContainer.style.gap = '2px';
            
            // Create piece image
            const pieceImg = document.createElement('img');
            pieceImg.src = `images/${pieceType}_black.svg`;
            pieceImg.className = 'piece-image';
            pieceImg.alt = pieceType;
            
            // Create count label
            const countLabel = document.createElement('span');
            countLabel.textContent = `×${count}`;
            countLabel.style.fontSize = '12px';
            countLabel.style.fontWeight = 'bold';
            countLabel.style.color = '#4ecdc4';
            
            pieceContainer.appendChild(pieceImg);
            pieceContainer.appendChild(countLabel);
            element.appendChild(pieceContainer);
        });
    }

    updateRecentMoves(moves) {
        const movesList = document.getElementById('moves-list');
        movesList.innerHTML = '';
        
        if (moves.length === 0) {
            movesList.innerHTML = '<div style="text-align: center; opacity: 0.5; padding: 20px;">No moves yet...</div>';
            return;
        }
        
        moves.reverse().forEach((move, index) => {
            const moveElement = document.createElement('div');
            moveElement.className = `move-item ${index === 0 ? 'latest' : ''}`;
            moveElement.textContent = move;
            movesList.appendChild(moveElement);
        });
    }

    updateHotSquares(hotSquares) {
        const element = document.getElementById('hot-squares');
        element.innerHTML = '';
        
        if (hotSquares.length === 0) {
            element.innerHTML = '<span style="opacity: 0.5; font-size: 12px;">None</span>';
            return;
        }
        
        hotSquares.forEach(square => {
            const squareElement = document.createElement('span');
            squareElement.style.background = 'rgba(255, 235, 59, 0.3)';
            squareElement.style.padding = '4px 8px';
            squareElement.style.borderRadius = '4px';
            squareElement.style.fontSize = '12px';
            squareElement.style.fontWeight = 'bold';
            squareElement.textContent = square;
            element.appendChild(squareElement);
        });
    }
    
    renderMetricsTable(metricsTable) {
        const container = document.getElementById('metrics-table');
        if (!container) return;
        
        if (!metricsTable || metricsTable.length === 0) {
            container.innerHTML = '<div style="opacity: 0.5; font-size: 12px; text-align: center;">No metrics data available</div>';
            return;
        }
        
        // Accumulate historical data for trend charts
        this.accumulateHistoricalData(metricsTable);
        
        // Get or create the table structure (preserve existing to avoid destroying charts)
        let table = document.getElementById('metrics-table-grid');
        let isNewTable = false;
        
        if (!table) {
            isNewTable = true;
            container.innerHTML = ''; // Only clear if creating new table
            table = document.createElement('div');
            table.id = 'metrics-table-grid';
            table.style.display = 'grid';
            table.style.gridTemplateColumns = '2fr 1fr 1fr 1fr';
            table.style.gap = '5px';
            table.style.fontSize = '10px';
            
            // Add header only for new table
            const headers = ['Metric', 'Last', 'Prev', 'Avg(5)'];
            headers.forEach((header, index) => {
                const headerCell = document.createElement('div');
                headerCell.className = 'metrics-header';
                headerCell.textContent = header;
                headerCell.style.fontWeight = 'bold';
                headerCell.style.color = '#4ecdc4';
                headerCell.style.padding = '4px';
                headerCell.style.backgroundColor = 'rgba(0,0,0,0.3)';
                headerCell.style.borderRadius = '3px';
                table.appendChild(headerCell);
            });
            
            container.appendChild(table);
        }
        
        // Track current metrics for cleanup
        const currentMetrics = new Set(metricsTable.map(m => m.name));
        
        // Remove charts for metrics that no longer exist
        if (this.miniCharts) {
            for (const [metricName, chart] of this.miniCharts.entries()) {
                if (!currentMetrics.has(metricName)) {
                    chart.destroy();
                    this.miniCharts.delete(metricName);
                    
                    // Remove DOM elements for this metric
                    const elements = table.querySelectorAll(`[data-metric="${metricName}"]`);
                    elements.forEach(el => el.remove());
                }
            }
        }
        
        // Update or create metrics rows - append elements in correct order for each metric
        metricsTable.forEach((metric, index) => {
            const metricName = metric.name;
            
            // Find existing metric elements or create new ones
            let nameCell = table.querySelector(`[data-metric="${metricName}"][data-type="name"]`);
            let lastCell = table.querySelector(`[data-metric="${metricName}"][data-type="last"]`);
            let prevCell = table.querySelector(`[data-metric="${metricName}"][data-type="prev"]`);
            let avgCell = table.querySelector(`[data-metric="${metricName}"][data-type="avg"]`);
            let trendContainer = table.querySelector(`[data-metric="${metricName}"][data-type="trend"]`);
            
            // Create complete row elements for new metrics (append all at once in order)
            const isNewMetric = !nameCell;
            
            if (isNewMetric) {
                // Create all elements for this metric row
                nameCell = document.createElement('div');
                nameCell.className = 'metrics-cell';
                nameCell.setAttribute('data-metric', metricName);
                nameCell.setAttribute('data-type', 'name');
                nameCell.style.padding = '4px';
                nameCell.style.backgroundColor = 'rgba(255,255,255,0.05)';
                nameCell.style.borderRadius = '3px';
                
                lastCell = document.createElement('div');
                lastCell.className = 'metrics-cell';
                lastCell.setAttribute('data-metric', metricName);
                lastCell.setAttribute('data-type', 'last');
                lastCell.style.padding = '4px';
                lastCell.style.textAlign = 'center';
                lastCell.style.backgroundColor = 'rgba(255,255,255,0.05)';
                lastCell.style.borderRadius = '3px';
                
                prevCell = document.createElement('div');
                prevCell.className = 'metrics-cell';
                prevCell.setAttribute('data-metric', metricName);
                prevCell.setAttribute('data-type', 'prev');
                prevCell.style.padding = '4px';
                prevCell.style.textAlign = 'center';
                prevCell.style.backgroundColor = 'rgba(255,255,255,0.05)';
                prevCell.style.borderRadius = '3px';
                
                avgCell = document.createElement('div');
                avgCell.className = 'metrics-cell';
                avgCell.setAttribute('data-metric', metricName);
                avgCell.setAttribute('data-type', 'avg');
                avgCell.style.padding = '4px';
                avgCell.style.textAlign = 'center';
                avgCell.style.backgroundColor = 'rgba(255,255,255,0.05)';
                avgCell.style.borderRadius = '3px';
                
                // Append metric cells in correct order
                table.appendChild(nameCell);
                table.appendChild(lastCell);
                table.appendChild(prevCell);
                table.appendChild(avgCell);
                
                // CRITICAL FIX: Always create trend container immediately for proper DOM order
                trendContainer = document.createElement('div');
                trendContainer.className = 'metrics-trend';
                trendContainer.setAttribute('data-metric', metricName);
                trendContainer.setAttribute('data-type', 'trend');
                trendContainer.style.gridColumn = '1 / -1';
                trendContainer.style.height = '44px';
                trendContainer.style.marginBottom = '3px';
                trendContainer.style.backgroundColor = 'rgba(0,0,0,0.2)';
                trendContainer.style.borderRadius = '4px';
                trendContainer.style.border = '1px solid rgba(76, 205, 196, 0.3)';
                table.appendChild(trendContainer); // Add immediately after metric cells
            }
            
            // Update cell contents (this doesn't destroy charts)
            nameCell.textContent = metric.name;
            lastCell.textContent = this.formatMetricValue(metric.last);
            prevCell.textContent = this.formatMetricValue(metric.previous);
            avgCell.textContent = this.formatMetricValue(metric.average_5);
            
            // Handle trend chart updates
            if (!trendContainer) {
                trendContainer = table.querySelector(`[data-metric="${metricName}"][data-type="trend"]`);
            }
            
            // CRITICAL FIX: Always attempt to update mini chart with available data
            if (trendContainer) {
                const historicalValues = this.historicalData.metrics.get(metric.name);
                if (historicalValues && historicalValues.length > 0 && this.featuresEnabled.miniCharts) {
                    this.updateMiniChart(trendContainer, historicalValues, metric.name);
                } else {
                    // Show placeholder for charts without data
                    trendContainer.innerHTML = `<div style="text-align: center; color: #4ecdc4; font-size: 9px; line-height: 42px; opacity: 0.7;">Trend data loading...</div>`;
                }
            }
        });
    }
    
    updateMiniChart(container, data, metricName) {
        if (!container || !data || data.length === 0) return;
        
        // Skip mini charts if performance is degraded or disabled for clean UI
        if (this.isLowPerformance || !this.featuresEnabled.miniCharts) {
            container.innerHTML = '<div style="text-align: center; color: #4ecdc4; font-size: 9px; line-height: 36px; opacity: 0.7;">Trend: ' + (data[data.length-1] || 0).toFixed(3) + '</div>';
            return;
        }
        
        // CRITICAL FIX: Properly destroy existing chart before creating new one
        const existingChart = this.miniCharts && this.miniCharts.get(metricName);
        if (existingChart) {
            try {
                // Update existing chart with new data
                existingChart.data.labels = data.map((_, i) => i);
                existingChart.data.datasets[0].data = data;
                existingChart.update('none'); // No animation to prevent jittering
                return;
            } catch (error) {
                // CRITICAL: Properly clean up broken chart
                this._destroyMiniChart(metricName);
            }
        }
        
        // Create new chart only after cleanup
        this.createMiniChartInContainer(container, data, metricName);
    }
    
    _destroyMiniChart(metricName) {
        const existingChart = this.miniCharts && this.miniCharts.get(metricName);
        if (existingChart) {
            try {
                existingChart.destroy();
            } catch (error) {
                if (this.featuresEnabled.excessiveLogging) {
                    console.warn('Failed to destroy chart:', error);
                }
            }
            this.miniCharts.delete(metricName);
        }
    }
    
    createMiniChartInContainer(container, data, metricName) {
        // Clear container and create new canvas
        container.innerHTML = '';
        
        if (!data || data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #4ecdc4; font-size: 9px; line-height: 42px; opacity: 0.7;">No trend data</div>';
            return;
        }
        
        // CRITICAL FIX: Create properly sized canvas with explicit dimensions
        const canvas = document.createElement('canvas');
        const chartId = `mini-chart-${metricName.replace(/[^a-zA-Z0-9]/g, '-')}`;
        canvas.id = chartId;
        
        // Set explicit canvas dimensions for Chart.js
        const containerWidth = container.offsetWidth || 300;
        const containerHeight = 40;
        
        canvas.width = containerWidth;
        canvas.height = containerHeight;
        canvas.style.width = '100%';
        canvas.style.height = '40px';
        canvas.style.display = 'block';
        
        container.appendChild(canvas);
        
        // CRITICAL FIX: Wait for canvas to be in DOM before creating chart
        setTimeout(() => {
            try {
                // Verify Chart.js is available
                if (typeof Chart === 'undefined') {
                    container.innerHTML = '<div style="text-align: center; color: #ff6b6b; font-size: 9px; line-height: 42px;">Chart.js not loaded</div>';
                    return;
                }
                
                const chart = new Chart(canvas, {
                    type: 'line',
                    data: {
                        labels: data.map((_, i) => i),
                        datasets: [{
                            data: data,
                            borderColor: '#4ecdc4',
                            backgroundColor: 'rgba(76, 205, 196, 0.2)',
                            borderWidth: 2,
                            fill: true,
                            tension: 0.4,
                            pointRadius: 0,
                            pointHoverRadius: 0
                        }]
                    },
                    options: {
                        responsive: false, // CRITICAL: Disable responsive for mini charts
                        maintainAspectRatio: false,
                        animation: false,
                        plugins: {
                            legend: { display: false },
                            tooltip: { enabled: false }
                        },
                        scales: {
                            x: { 
                                display: false,
                                grid: { display: false }
                            },
                            y: { 
                                display: false,
                                grid: { display: false }
                            }
                        },
                        elements: {
                            point: { radius: 0 }
                        },
                        layout: {
                            padding: {
                                left: 2,
                                right: 2,
                                top: 2,
                                bottom: 2
                            }
                        }
                    }
                });
                
                // Store the chart for future updates
                if (!this.miniCharts) {
                    this.miniCharts = new Map();
                }
                this.miniCharts.set(metricName, chart);
                
            } catch (error) {
                console.warn('Mini chart creation failed:', error);
                container.innerHTML = '<div style="text-align: center; color: #ff6b6b; font-size: 9px; line-height: 42px;">Chart error</div>';
            }
        }, 10); // Small delay to ensure DOM is ready
    }
    
    accumulateHistoricalData(currentMetrics) {
        const timestamp = Date.now();
        
        // Throttle historical data accumulation to 1 point per minute for meaningful trends
        if (timestamp - this.lastHistoryUpdate < this.historyUpdateInterval) {
            return; // Skip this update, too soon for trend data
        }
        this.lastHistoryUpdate = timestamp;
        
        // CRITICAL FIX: Enforce bounds BEFORE adding new data
        if (this.historicalData.timestamps.length >= this.maxHistoryLength) {
            const excessCount = this.historicalData.timestamps.length - this.maxHistoryLength + 1;
            this.historicalData.timestamps.splice(0, excessCount);
            
            // Atomic cleanup of all metric arrays
            this.historicalData.metrics.forEach(values => {
                if (values.length >= this.maxHistoryLength) {
                    values.splice(0, excessCount);
                }
            });
        }
        
        this.historicalData.timestamps.push(timestamp);
        
        // Add current values to historical data with bounds checking
        currentMetrics.forEach(metric => {
            if (!this.historicalData.metrics.has(metric.name)) {
                this.historicalData.metrics.set(metric.name, []);
            }
            
            const metricArray = this.historicalData.metrics.get(metric.name);
            metricArray.push(metric.last);
            
            // Immediate bounds enforcement per metric
            if (metricArray.length > this.maxHistoryLength) {
                metricArray.splice(0, metricArray.length - this.maxHistoryLength);
            }
        });
        
        // Periodic deep cleanup to prevent memory leaks
        if (this.historicalData.timestamps.length % 50 === 0) {
            this._performDeepHistoricalDataCleanup();
        }
    }
    
    accumulateWinRateHistory(winRate, lossRate, drawRate) {
        // Throttle win rate history to same interval as other metrics for consistent trends
        const timestamp = Date.now();
        if (timestamp - this.lastHistoryUpdate < this.historyUpdateInterval) {
            return; // Skip this update, maintain consistent timing with other metrics
        }
        
        // Initialize win rate metric arrays if they don't exist
        if (!this.historicalData.metrics.has('win_rate')) {
            this.historicalData.metrics.set('win_rate', []);
        }
        if (!this.historicalData.metrics.has('loss_rate')) {
            this.historicalData.metrics.set('loss_rate', []);
        }
        if (!this.historicalData.metrics.has('draw_rate')) {
            this.historicalData.metrics.set('draw_rate', []);
        }
        
        // Add current rates to historical data
        const winHistory = this.historicalData.metrics.get('win_rate');
        const lossHistory = this.historicalData.metrics.get('loss_rate');
        const drawHistory = this.historicalData.metrics.get('draw_rate');
        
        winHistory.push(winRate);
        lossHistory.push(lossRate);
        drawHistory.push(drawRate);
        
        // Enforce bounds for win rate data
        if (winHistory.length > this.maxHistoryLength) {
            winHistory.splice(0, winHistory.length - this.maxHistoryLength);
        }
        if (lossHistory.length > this.maxHistoryLength) {
            lossHistory.splice(0, lossHistory.length - this.maxHistoryLength);
        }
        if (drawHistory.length > this.maxHistoryLength) {
            drawHistory.splice(0, drawHistory.length - this.maxHistoryLength);
        }
    }
    
    _performDeepHistoricalDataCleanup() {
        // Remove orphaned metrics that no longer exist
        const activeMetrics = new Set();
        
        // Always keep win rate metrics as they're essential
        activeMetrics.add('win_rate');
        activeMetrics.add('loss_rate');
        activeMetrics.add('draw_rate');
        
        // Only keep metrics that have been updated recently
        this.historicalData.metrics.forEach((values, metricName) => {
            if (values.length > 0) {
                activeMetrics.add(metricName);
            }
        });
        
        // Clean up unused metric arrays
        for (const [metricName, values] of this.historicalData.metrics.entries()) {
            if (values.length === 0 && !activeMetrics.has(metricName)) {
                this.historicalData.metrics.delete(metricName);
            }
        }
        
        // Force bounds compliance
        this.historicalData.timestamps = this.historicalData.timestamps.slice(-this.maxHistoryLength);
        this.historicalData.metrics.forEach((values, metricName) => {
            if (values.length > this.maxHistoryLength) {
                this.historicalData.metrics.set(metricName, values.slice(-this.maxHistoryLength));
            }
        });
    }
    
    
    formatMetricValue(value) {
        if (value === null || value === undefined) {
            return 'N/A';
        }
        if (typeof value === 'number') {
            if (Math.abs(value) >= 1000) {
                return value.toExponential(2);
            } else if (Math.abs(value) < 0.001) {
                return value.toExponential(3);
            } else {
                return value.toFixed(3);
            }
        }
        return String(value);
    }

    squareNameToCoords(squareName) {
        // Convert square name like "5e" to board coordinates
        // This is a simplified conversion - may need adjustment based on actual format
        const match = squareName.match(/(\d)([a-i])/);
        if (match) {
            const col = parseInt(match[1]) - 1;
            const row = match[2].charCodeAt(0) - 'a'.charCodeAt(0);
            return { row, col };
        }
        return null;
    }

    formatNumber(num) {
        if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toString();
    }
    
    safeUpdateElement(elementId, value) {
        try {
            const element = document.getElementById(elementId);
            if (element) {
                element.textContent = value;
            } else {
                console.warn(`⚠️ Element '${elementId}' not found - value: ${value}`);
            }
        } catch (error) {
            console.error(`❌ Error updating element '${elementId}':`, error);
        }
    }
    
    showErrorIndicator(message) {
        try {
            // Show a temporary error message in the processing indicator
            const indicator = document.getElementById('processing-indicator');
            if (indicator) {
                indicator.textContent = `⚠️ ${message}`;
                indicator.style.display = 'block';
                indicator.style.background = 'rgba(255, 107, 107, 0.2)';
                indicator.style.color = '#ff6b6b';
                
                // Hide after 5 seconds
                setTimeout(() => {
                    indicator.style.display = 'none';
                    indicator.style.background = 'rgba(255, 193, 7, 0.2)';
                    indicator.style.color = '#ffc107';
                    indicator.textContent = '🧠 AI is learning...';
                }, 5000);
            }
        } catch (error) {
            console.error('❌ Error showing error indicator:', error);
        }
    }

    updateAdvancedVisualizations(data) {
        try {
            if (!window.advancedVisualizations) {
                return; // Silent fail - advanced visualizations are optional
            }
        
            // Extract and update neural confidence data from policy output
            if (data.policy_confidence) {
                window.advancedVisualizations.updateNeuralConfidence(data.policy_confidence);
            }
            
            // Update exploration gauge with entropy values
            if (data.entropy !== undefined) {
                window.advancedVisualizations.updateExploration(data.entropy);
            }
            
            // Update advantage oscillation with position evaluation
            if (data.value_estimate !== undefined) {
                window.advancedVisualizations.updateAdvantage(data.value_estimate);
            }
            
            // Update skill radar with learning progress
            if (data.skill_metrics) {
                window.advancedVisualizations.updateSkills(data.skill_metrics);
            }
            
            // Update gradient flow visualization
            if (data.gradient_norms) {
                window.advancedVisualizations.updateGradients(data.gradient_norms);
            }
            
            // Update buffer dynamics
            if (data.buffer_info) {
                window.advancedVisualizations.updateBuffer({
                    size: data.buffer_info.buffer_size,
                    capacity: data.buffer_info.buffer_capacity,
                    qualityDistribution: data.buffer_info.quality_distribution || []
                });
            }
        } catch (error) {
            if (this.featuresEnabled.excessiveLogging) {
                console.error('❌ Error in updateAdvancedVisualizations:', error);
            }
            // Advanced visualizations are optional - don't show error indicator
        }
    }
}

// Memory Management System for Browser Stability
class MemoryManager {
    constructor(webui) {
        this.webui = webui;
        this.memoryCheckInterval = 30000; // Check every 30 seconds
        this.warningThreshold = 50 * 1024 * 1024; // 50MB warning
        this.criticalThreshold = 100 * 1024 * 1024; // 100MB critical
        this.lastCleanup = Date.now();
        this.cleanupInterval = 60000; // Force cleanup every minute
        
        this.startMemoryMonitoring();
    }
    
    startMemoryMonitoring() {
        setInterval(() => {
            this.checkMemoryUsage();
            this.performPeriodicCleanup();
        }, this.memoryCheckInterval);
        
        // Listen for page unload to cleanup resources
        window.addEventListener('beforeunload', () => {
            this.forceCleanup();
        });
    }
    
    checkMemoryUsage() {
        if (performance.memory) {
            const used = performance.memory.usedJSHeapSize;
            const total = performance.memory.totalJSHeapSize;
            
            if (used > this.criticalThreshold) {
                this.triggerEmergencyCleanup();
            } else if (used > this.warningThreshold) {
                this.triggerModerateCleanup();
            }
            
            // Log memory stats periodically
            if (this.webui.featuresEnabled.excessiveLogging) {
                console.log(`Memory: ${(used / 1024 / 1024).toFixed(1)}MB / ${(total / 1024 / 1024).toFixed(1)}MB`);
            }
        }
    }
    
    triggerEmergencyCleanup() {
        console.warn('🚨 EMERGENCY: Critical memory usage detected - triggering aggressive cleanup');
        
        // Disable all non-essential features
        this.webui.isLowPerformance = true;
        this.webui.featuresEnabled.miniCharts = false;
        this.webui.featuresEnabled.animations = false;
        this.webui.featuresEnabled.advancedVisualizations = false;
        
        this.forceCleanup();
        
        // Show user warning
        this.webui.showErrorIndicator('High memory usage - some features disabled');
    }
    
    triggerModerateCleanup() {
        console.warn('⚠️ WARNING: High memory usage detected - starting cleanup');
        
        // Reduce update frequency
        this.webui.uiUpdateInterval = 200; // Slow down updates
        
        this.performPeriodicCleanup();
    }
    
    performPeriodicCleanup() {
        const now = Date.now();
        if (now - this.lastCleanup < this.cleanupInterval) return;
        
        this.lastCleanup = now;
        
        // Clean up historical data
        this.webui._performDeepHistoricalDataCleanup();
        
        // Clean up orphaned chart instances
        this.cleanupOrphanedCharts();
        
        // Trigger garbage collection if available
        if (window.gc) {
            try {
                window.gc();
            } catch (e) {
                // Ignore - gc() is not always available
            }
        }
    }
    
    cleanupOrphanedCharts() {
        if (!this.webui.miniCharts) return;
        
        const orphanedCharts = [];
        
        this.webui.miniCharts.forEach((chart, metricName) => {
            if (!chart.canvas || !chart.canvas.parentNode) {
                orphanedCharts.push(metricName);
            }
        });
        
        orphanedCharts.forEach(metricName => {
            this.webui._destroyMiniChart(metricName);
        });
        
        if (orphanedCharts.length > 0 && this.webui.featuresEnabled.excessiveLogging) {
            console.log(`Cleaned up ${orphanedCharts.length} orphaned charts`);
        }
    }
    
    forceCleanup() {
        // Destroy all charts
        if (this.webui.miniCharts) {
            this.webui.miniCharts.forEach((chart, metricName) => {
                try {
                    chart.destroy();
                } catch (e) {
                    // Ignore errors during cleanup
                }
            });
            this.webui.miniCharts.clear();
        }
        
        // Clear historical data
        this.webui.historicalData.timestamps = [];
        this.webui.historicalData.metrics.clear();
        
        // Clear any queued updates
        if (this.webui.updateThrottler) {
            this.webui.updateThrottler.clearQueue();
        }
    }
}

// Performance Monitoring System
class PerformanceMonitor {
    constructor(webui) {
        this.webui = webui;
        this.frameCount = 0;
        this.lastFrameCheck = Date.now();
        this.fpsThreshold = 15; // FPS below this triggers performance mode
        
        this.startPerformanceMonitoring();
    }
    
    startPerformanceMonitoring() {
        setInterval(() => {
            this.checkFrameRate();
        }, 5000); // Check every 5 seconds
    }
    
    checkFrameRate() {
        const now = Date.now();
        const elapsed = now - this.lastFrameCheck;
        const fps = (this.frameCount * 1000) / elapsed;
        
        if (fps < this.fpsThreshold && fps > 0) {
            if (!this.webui.isLowPerformance) {
                console.warn(`Low FPS detected: ${fps.toFixed(1)} - enabling performance mode`);
                this.enablePerformanceMode();
            }
        } else if (fps > this.fpsThreshold * 1.5 && this.webui.isLowPerformance) {
            console.log(`FPS recovered: ${fps.toFixed(1)} - disabling performance mode`);
            this.disablePerformanceMode();
        }
        
        this.frameCount = 0;
        this.lastFrameCheck = now;
    }
    
    recordFrame() {
        this.frameCount++;
    }
    
    enablePerformanceMode() {
        this.webui.isLowPerformance = true;
        this.webui.uiUpdateInterval = 200; // Reduce update frequency
        this.webui.featuresEnabled.animations = false;
        
        // Show performance mode indicator
        this.webui.showErrorIndicator('Performance mode enabled - some features disabled');
    }
    
    disablePerformanceMode() {
        this.webui.isLowPerformance = false;
        this.webui.uiUpdateInterval = 100; // Restore normal frequency
        this.webui.featuresEnabled.animations = true;
        this.webui.featuresEnabled.miniCharts = true;
        this.webui.featuresEnabled.advancedVisualizations = true;
    }
}

// Update Throttling System
class UpdateThrottler {
    constructor() {
        this.updateQueue = [];
        this.isProcessing = false;
        this.maxQueueSize = 10; // Prevent queue from growing too large
    }
    
    scheduleUpdate(updateFunction) {
        if (this.updateQueue.length >= this.maxQueueSize) {
            // Drop oldest updates if queue is full
            this.updateQueue.shift();
        }
        
        this.updateQueue.push(updateFunction);
        
        if (!this.isProcessing) {
            this.processQueue();
        }
    }
    
    processQueue() {
        this.isProcessing = true;
        
        const processNext = () => {
            if (this.updateQueue.length === 0) {
                this.isProcessing = false;
                return;
            }
            
            const updateFunction = this.updateQueue.shift();
            try {
                updateFunction();
            } catch (error) {
                console.error('Error in throttled update:', error);
            }
            
            // Use requestAnimationFrame for smooth processing
            requestAnimationFrame(processNext);
        };
        
        requestAnimationFrame(processNext);
    }
    
    clearQueue() {
        this.updateQueue = [];
        this.isProcessing = false;
    }
}

// Initialize the WebUI when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.keiseiWebUI = new KeiseiWebUI();
    
    // Initialize advanced visualizations
    if (typeof AdvancedVisualizationManager !== 'undefined') {
        window.advancedVisualizations = new AdvancedVisualizationManager();
        console.log('✅ AdvancedVisualizationManager initialized');
    } else {
        console.warn('⚠️ AdvancedVisualizationManager not found');
    }
});