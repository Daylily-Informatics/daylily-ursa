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

const pricingChartState = {
    chart: null,
    region: 'us-west-2',
    partitions: ['i8', 'i128', 'i192', 'i192mem', 'i192bigmem'],
};

const azPalette = ['#0f3460', '#00d9a6', '#e94560', '#ffc107', '#4f46e5', '#f97316'];

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

function formatSnapshotTimestamp(timestamp) {
    if (!timestamp) return '';
    return new Date(timestamp).toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: 'numeric',
        minute: '2-digit',
    });
}

function buildPricingDatasets(snapshotPayload, selectedRegion, selectedPartitions) {
    const snapshots = (snapshotPayload.snapshots || []).filter(item => item.region === selectedRegion);
    const labels = [];
    const azNames = new Set();
    const boxLookup = {};
    const pointLookup = {};

    snapshots.forEach(snapshot => {
        (snapshot.partitions || []).forEach(partitionEntry => {
            if (!selectedPartitions.includes(partitionEntry.partition)) return;
            const label = `${formatSnapshotTimestamp(snapshot.captured_at)}\n${partitionEntry.partition}`;
            labels.push(label);

            (partitionEntry.availability_zones || []).forEach(zoneEntry => {
                azNames.add(zoneEntry.availability_zone);
                if (!boxLookup[zoneEntry.availability_zone]) {
                    boxLookup[zoneEntry.availability_zone] = {};
                    pointLookup[zoneEntry.availability_zone] = [];
                }
                boxLookup[zoneEntry.availability_zone][label] = zoneEntry.box;
                (zoneEntry.points || []).forEach(point => {
                    pointLookup[zoneEntry.availability_zone].push({
                        x: label,
                        y: point.vcpu_cost_per_hour,
                        instanceType: point.instance_type,
                        hourlySpotPrice: point.hourly_spot_price,
                    });
                });
            });
        });
    });

    const azList = Array.from(azNames).sort();
    const datasets = [];

    azList.forEach((azName, index) => {
        const color = azPalette[index % azPalette.length];
        datasets.push({
            type: 'boxplot',
            label: azName,
            data: labels.map(label => boxLookup[azName][label] || null),
            backgroundColor: `${color}33`,
            borderColor: color,
            borderWidth: 1.5,
            itemRadius: 0,
        });
        datasets.push({
            type: 'scatter',
            label: `${azName} instances`,
            data: pointLookup[azName],
            pointRadius: 4,
            pointHoverRadius: 5,
            pointBackgroundColor: color,
            pointBorderColor: '#ffffff',
            pointBorderWidth: 1,
            showLine: false,
        });
    });

    return { labels, datasets };
}

function renderCheapestAzSummary(snapshotPayload, selectedRegion, summaryEl) {
    if (!summaryEl) return;
    const entries = (snapshotPayload.latest_cheapest_az || []).filter(item => item.region === selectedRegion);
    if (!entries.length) {
        summaryEl.textContent = 'No pricing history captured yet.';
        return;
    }
    summaryEl.innerHTML = entries
        .sort((a, b) => a.partition.localeCompare(b.partition))
        .map(item => `${item.partition}: <code>${item.availability_zone}</code> @ $${item.median_vcpu_cost_per_hour.toFixed(4)}/vCPU-hr`)
        .join(' | ');
}

async function loadPricingSnapshotChart(state, els) {
    const params = new URLSearchParams();
    params.set('region', state.region);
    params.set('partitions', state.partitions.join(','));

    const response = await fetch(`/api/pricing-snapshots?${params.toString()}`);
    if (!response.ok) {
        throw new Error(`Failed to load pricing snapshots (${response.status})`);
    }

    const payload = await response.json();
    const latestRun = (payload.runs || [])[0];
    if (els.statusEl) {
        if (latestRun?.snapshot_captured_at) {
            const prefix = latestRun.status === 'running' ? 'Capture running.' : 'Last capture';
            els.statusEl.textContent = `${prefix} ${formatSnapshotTimestamp(latestRun.snapshot_captured_at)}`;
        } else if (latestRun) {
            els.statusEl.textContent = `Latest pricing job is ${latestRun.status}.`;
        } else {
            els.statusEl.textContent = 'No pricing snapshots captured yet.';
        }
    }

    renderCheapestAzSummary(payload, state.region, els.summaryEl);
    const { labels, datasets } = buildPricingDatasets(payload, state.region, state.partitions);

    if (state.chart) {
        state.chart.destroy();
    }

    state.chart = new Chart(els.canvas, {
        data: {
            labels,
            datasets,
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    labels: {
                        filter: item => !String(item.text || '').endsWith('instances'),
                    },
                },
                tooltip: {
                    callbacks: {
                        label(context) {
                            if (context.dataset.type === 'scatter') {
                                const raw = context.raw || {};
                                return `${raw.instanceType}: $${Number(raw.y || 0).toFixed(4)}/vCPU-hr`;
                            }
                            const raw = context.raw || {};
                            return `median $${Number(raw.median || 0).toFixed(4)} (min ${Number(raw.min || 0).toFixed(4)}, max ${Number(raw.max || 0).toFixed(4)})`;
                        },
                    },
                },
            },
            scales: {
                x: {
                    ticks: {
                        callback(value, index) {
                            const label = labels[index] || '';
                            return label.split('\n');
                        },
                    },
                },
                y: {
                    title: {
                        display: true,
                        text: 'vCPU Cost / Hour (USD)',
                    },
                },
            },
        },
    });
}

async function triggerPricingSnapshot(els) {
    const response = await fetch('/api/pricing-snapshots/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        throw new Error(payload.detail || payload.error || 'Failed to queue pricing snapshot');
    }
    if (els.statusEl) {
        els.statusEl.textContent = `Pricing capture queued (run ${payload.run_id}). Refreshing shortly…`;
    }
}

async function initPricingSnapshotChart(options) {
    const canvas = document.getElementById(options.canvasId);
    if (!canvas) return null;

    const statusEl = document.getElementById(options.statusId);
    const summaryEl = document.getElementById(options.summaryId);
    const runButtonEl = options.runButtonId ? document.getElementById(options.runButtonId) : null;
    const regionTabsEl = document.getElementById(options.regionTabsId);
    const partitionFiltersEl = document.getElementById(options.partitionFiltersId);
    const els = { canvas, statusEl, summaryEl };

    async function refresh() {
        try {
            await loadPricingSnapshotChart(pricingChartState, els);
        } catch (error) {
            console.error('Failed to render pricing snapshot chart:', error);
            if (statusEl) {
                statusEl.textContent = error.message || 'Failed to load pricing data';
            }
        }
    }

    if (regionTabsEl) {
        regionTabsEl.querySelectorAll('.pricing-region-tab').forEach(tab => {
            tab.addEventListener('click', async () => {
                pricingChartState.region = tab.dataset.region || 'us-west-2';
                regionTabsEl.querySelectorAll('.pricing-region-tab').forEach(el => el.classList.remove('active'));
                tab.classList.add('active');
                await refresh();
            });
        });
    }

    if (partitionFiltersEl) {
        partitionFiltersEl.querySelectorAll('.pricing-partition-chip').forEach(chip => {
            chip.addEventListener('click', async () => {
                chip.classList.toggle('active');
                const selected = Array.from(partitionFiltersEl.querySelectorAll('.pricing-partition-chip.active'))
                    .map(el => el.dataset.partition)
                    .filter(Boolean);
                pricingChartState.partitions = selected.length
                    ? selected
                    : ['i8', 'i128', 'i192', 'i192mem', 'i192bigmem'];
                if (selected.length === 0) {
                    partitionFiltersEl.querySelectorAll('.pricing-partition-chip').forEach(el => el.classList.add('active'));
                }
                await refresh();
            });
        });
    }

    if (runButtonEl && options.isAdmin) {
        runButtonEl.addEventListener('click', async () => {
            const originalHtml = runButtonEl.innerHTML;
            runButtonEl.disabled = true;
            runButtonEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Running…';
            try {
                await triggerPricingSnapshot(els);
                setTimeout(refresh, 3000);
            } catch (error) {
                console.error('Failed to queue pricing snapshot:', error);
                if (statusEl) {
                    statusEl.textContent = error.message || 'Failed to queue pricing snapshot';
                }
            } finally {
                runButtonEl.disabled = false;
                runButtonEl.innerHTML = originalHtml;
            }
        });
    }

    await refresh();
    return pricingChartState.chart;
}
