/**
 * Ursa Customer Portal - Charts
 */

// Chart.js default configuration
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.color = '#6c757d';

// Color palette
const chartColors = {
    primary: '#0f3460',
    accent: '#00d9a6',
    secondary: '#1a1a2e',
    warning: '#ffc107',
    error: '#e94560',
    success: '#00d9a6',
    gray: '#6c757d',
};

// Initialize activity chart (dashboard) - fetches real data from API
async function initActivityChart(canvasId, customerId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    // Show loading state
    const container = ctx.parentElement;
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chart-loading';
    loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading activity data...';
    loadingDiv.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:var(--text-muted);';
    container.style.position = 'relative';
    container.appendChild(loadingDiv);

    let labels = [];
    let submittedData = [];
    let completedData = [];
    let failedData = [];

    // Fetch real data if customerId provided
    if (customerId) {
        try {
            const response = await fetch(`/api/customers/${customerId}/dashboard/activity?days=30`);
            if (response.ok) {
                const data = await response.json();
                labels = data.labels || [];
                submittedData = data.datasets?.submitted || [];
                completedData = data.datasets?.completed || [];
                failedData = data.datasets?.failed || [];
            }
        } catch (error) {
            console.error('Failed to fetch activity data:', error);
        }
    }

    // Fall back to empty data if no data fetched
    if (labels.length === 0) {
        for (let i = 29; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            submittedData.push(0);
            completedData.push(0);
            failedData.push(0);
        }
    }

    loadingDiv.remove();

    const chart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Submitted',
                    data: submittedData,
                    borderColor: chartColors.primary,
                    backgroundColor: 'rgba(15, 52, 96, 0.1)',
                    fill: true,
                    tension: 0.4,
                },
                {
                    label: 'Completed',
                    data: completedData,
                    borderColor: chartColors.accent,
                    backgroundColor: 'rgba(0, 217, 166, 0.1)',
                    fill: true,
                    tension: 0.4,
                },
                {
                    label: 'Failed',
                    data: failedData,
                    borderColor: chartColors.error,
                    backgroundColor: 'rgba(233, 69, 96, 0.1)',
                    fill: true,
                    tension: 0.4,
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1,
                    },
                },
            },
        },
    });

    return chart;
}

// Initialize cost chart (usage page) - fetches real data from API
async function initCostChart(canvasId, customerId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    // Show loading state
    const container = ctx.parentElement;
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chart-loading';
    loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading cost data...';
    loadingDiv.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:var(--text-muted);';
    container.style.position = 'relative';
    container.appendChild(loadingDiv);

    let labels = [];
    let costData = [];

    // Fetch real data if customerId provided
    if (customerId) {
        try {
            const response = await fetch(`/api/customers/${customerId}/dashboard/cost-history?days=30`);
            if (response.ok) {
                const data = await response.json();
                labels = data.labels || [];
                costData = data.costs || [];
            }
        } catch (error) {
            console.error('Failed to fetch cost history:', error);
        }
    }

    // Fall back to empty data if no data fetched
    if (labels.length === 0) {
        for (let i = 29; i >= 0; i--) {
            const date = new Date();
            date.setDate(date.getDate() - i);
            labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
            costData.push(0);
        }
    }

    loadingDiv.remove();

    const chart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Daily Cost ($)',
                data: costData,
                backgroundColor: chartColors.accent,
                borderRadius: 4,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: value => '$' + value,
                    },
                },
            },
        },
    });

    return chart;
}

// Initialize breakdown chart (usage page) - fetches real data from API
async function initBreakdownChart(canvasId, customerId) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;

    // Show loading state
    const container = ctx.parentElement;
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'chart-loading';
    loadingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Loading breakdown...';
    loadingDiv.style.cssText = 'position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);color:var(--text-muted);';
    container.style.position = 'relative';
    container.appendChild(loadingDiv);

    let categories = ['Compute'];
    let values = [0];

    // Fetch real data if customerId provided
    if (customerId) {
        try {
            const response = await fetch(`/api/customers/${customerId}/dashboard/cost-breakdown`);
            if (response.ok) {
                const data = await response.json();
                categories = data.categories || ['Compute'];
                values = data.values || [0];
            }
        } catch (error) {
            console.error('Failed to fetch cost breakdown:', error);
        }
    }

    loadingDiv.remove();

    // Generate colors based on number of categories
    const backgroundColors = [
        chartColors.primary,
        chartColors.accent,
        chartColors.warning,
        chartColors.gray,
        chartColors.error,
    ].slice(0, categories.length);

    const chart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: categories,
            datasets: [{
                data: values,
                backgroundColor: backgroundColors,
                borderWidth: 0,
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                },
            },
            cutout: '60%',
        },
    });

    return chart;
}

// Update chart with real data (generic helper)
async function updateChartData(chart, customerId, endpoint) {
    try {
        const data = await UrsaAPI.get(endpoint);
        chart.data.labels = data.labels;
        chart.data.datasets[0].data = data.values;
        chart.update();
    } catch (error) {
        console.error('Failed to update chart:', error);
    }
}

