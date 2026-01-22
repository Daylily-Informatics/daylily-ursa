/**
 * Daylily Customer Portal - Worksets Management
 */

// Debounce timer for search input
let searchDebounceTimer = null;

// Filter worksets - navigates to URL with filter parameters for server-side filtering
function filterWorksets(isSearchInput = false) {
    const status = document.getElementById('filter-status')?.value || '';
    const type = document.getElementById('filter-type')?.value || '';
    const search = document.getElementById('search-worksets')?.value || '';
    const sort = document.getElementById('filter-sort')?.value || 'created_desc';

    // Build URL with filter parameters
    const params = new URLSearchParams();
    params.set('page', '1');  // Always reset to page 1 when filtering
    if (status) params.set('status', status);
    if (type) params.set('type', type);
    if (search.trim()) params.set('search', search.trim());
    if (sort && sort !== 'created_desc') params.set('sort', sort);

    const newUrl = '/portal/worksets?' + params.toString();

    // For search input, debounce to avoid too many requests
    if (isSearchInput) {
        if (searchDebounceTimer) {
            clearTimeout(searchDebounceTimer);
        }
        searchDebounceTimer = setTimeout(() => {
            window.location.href = newUrl;
        }, 400);  // 400ms delay for typing
    } else {
        // For dropdown changes, navigate immediately
        window.location.href = newUrl;
    }
}

// Handle search input with debouncing
function handleSearchInput() {
    filterWorksets(true);
}

// Clear all filters
function clearFilters() {
    window.location.href = '/portal/worksets?page=1';
}

// Toggle select all
function toggleSelectAll() {
    const selectAll = document.getElementById('select-all');
    const checkboxes = document.querySelectorAll('.workset-checkbox');
    
    checkboxes.forEach(cb => {
        cb.checked = selectAll.checked;
    });
    
    updateBulkActions();
}

// Update bulk actions visibility
function updateBulkActions() {
    const checked = document.querySelectorAll('.workset-checkbox:checked');
    const bulkActions = document.getElementById('bulk-actions');
    const countEl = document.getElementById('selected-count');
    
    if (bulkActions) {
        bulkActions.classList.toggle('d-none', checked.length === 0);
    }
    if (countEl) {
        countEl.textContent = checked.length;
    }
}

// Clear selection
function clearSelection() {
    document.querySelectorAll('.workset-checkbox').forEach(cb => cb.checked = false);
    document.getElementById('select-all').checked = false;
    updateBulkActions();
}

// Refresh worksets list
async function refreshWorksets() {
    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;
    
    showLoading('Refreshing worksets...');
    
    try {
        const data = await DaylilyAPI.worksets.list(customerId);
        // Reload page to show updated data
        window.location.reload();
    } catch (error) {
        showToast('error', 'Refresh Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Cancel workset
async function cancelWorkset(worksetId) {
    if (!confirm('Are you sure you want to cancel this workset?')) return;
    
    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;
    
    showLoading('Cancelling workset...');
    
    try {
        await DaylilyAPI.worksets.cancel(customerId, worksetId);
        showToast('success', 'Workset Cancelled', 'The workset has been cancelled');
        setTimeout(() => window.location.reload(), 1000);
    } catch (error) {
        showToast('error', 'Cancel Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Retry workset
async function retryWorkset(worksetId) {
    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;
    
    showLoading('Retrying workset...');
    
    try {
        await DaylilyAPI.worksets.retry(customerId, worksetId);
        showToast('success', 'Workset Restarted', 'The workset has been queued for retry');
        setTimeout(() => window.location.reload(), 1000);
    } catch (error) {
        showToast('error', 'Retry Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Bulk cancel
async function bulkCancel() {
    const selected = Array.from(document.querySelectorAll('.workset-checkbox:checked')).map(cb => cb.value);
    if (selected.length === 0) return;

    if (!confirm(`Cancel ${selected.length} workset(s)?`)) return;

    showLoading('Cancelling worksets...');

    try {
        const customerId = window.UrsaConfig?.customerId;
        if (!customerId) return;

        let successCount = 0;
        for (const worksetId of selected) {
            try {
                await DaylilyAPI.worksets.cancel(customerId, worksetId);
                successCount++;
            } catch (e) {
                console.error(`Failed to cancel ${worksetId}:`, e);
            }
        }
        showToast('success', 'Worksets Cancelled', `${successCount} of ${selected.length} worksets cancelled`);
        setTimeout(() => window.location.reload(), 1500);
    } catch (error) {
        showToast('error', 'Bulk Cancel Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Bulk archive
async function bulkArchive() {
    const selected = getSelectedWorksets();
    if (selected.length === 0) {
        showToast('warning', 'No Selection', 'Please select worksets to archive');
        return;
    }

    // Filter out worksets that can't be archived
    const validStates = ['ready', 'completed', 'complete', 'error', 'failed'];
    const archivable = selected.filter(ws => {
        const row = document.querySelector(`tr[data-workset-id="${ws}"]`);
        const status = row?.dataset.status?.toLowerCase();
        return validStates.includes(status);
    });

    if (archivable.length === 0) {
        showToast('warning', 'Invalid Selection', 'Selected worksets cannot be archived (in-progress or already archived/deleted)');
        return;
    }

    if (archivable.length < selected.length) {
        const skipped = selected.length - archivable.length;
        if (!confirm(`${skipped} workset(s) will be skipped (in-progress or already archived/deleted).\n\nArchive ${archivable.length} workset(s)?`)) {
            return;
        }
    } else {
        if (!confirm(`Archive ${archivable.length} workset(s)?\n\nArchived worksets can be restored later.`)) {
            return;
        }
    }

    const reason = prompt('Enter reason for archiving (optional):');
    if (reason === null) return; // User cancelled

    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;

    showLoading(`Archiving ${archivable.length} worksets...`);

    try {
        let successCount = 0;
        let errors = [];

        for (const worksetId of archivable) {
            try {
                await DaylilyAPI.worksets.archive(customerId, worksetId, reason || undefined);
                successCount++;
            } catch (e) {
                console.error(`Failed to archive ${worksetId}:`, e);
                errors.push(worksetId);
            }
        }

        if (successCount > 0) {
            showToast('success', 'Bulk Archive Complete', `${successCount} of ${archivable.length} worksets archived`);
        }
        if (errors.length > 0) {
            showToast('warning', 'Some Failed', `${errors.length} worksets failed to archive`);
        }

        setTimeout(() => window.location.reload(), 1500);
    } catch (error) {
        showToast('error', 'Bulk Archive Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Bulk delete
async function bulkDelete() {
    const selected = getSelectedWorksets();
    if (selected.length === 0) {
        showToast('warning', 'No Selection', 'Please select worksets to delete');
        return;
    }

    // Filter out worksets that can't be deleted
    const validStates = ['ready', 'completed', 'complete', 'error', 'failed', 'archived'];
    const deletable = selected.filter(ws => {
        const row = document.querySelector(`tr[data-workset-id="${ws}"]`);
        const status = row?.dataset.status?.toLowerCase();
        return validStates.includes(status);
    });

    if (deletable.length === 0) {
        showToast('warning', 'Invalid Selection', 'Selected worksets cannot be deleted (in-progress)');
        return;
    }

    // First confirmation: soft or hard delete
    const hardDelete = confirm(
        `Delete ${deletable.length} workset(s)?\n\n` +
        `Choose deletion type:\n` +
        `• OK = PERMANENT DELETE (removes all S3 data - CANNOT BE UNDONE)\n` +
        `• Cancel = Soft delete (marks as deleted, data preserved)`
    );

    // Second confirmation for hard delete
    if (hardDelete) {
        const finalConfirm = confirm(
            `⚠️ FINAL WARNING ⚠️\n\n` +
            `You are about to PERMANENTLY DELETE ${deletable.length} workset(s) and ALL their data from S3.\n\n` +
            `This action CANNOT be undone!\n\n` +
            `Are you absolutely sure?`
        );
        if (!finalConfirm) return;
    } else {
        if (!confirm(`Soft delete ${deletable.length} workset(s)?\n\nData will be preserved and can be recovered.`)) {
            return;
        }
    }

    const reason = prompt('Enter reason for deletion (optional):');
    if (reason === null) return; // User cancelled

    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;

    const deleteType = hardDelete ? 'permanently deleting' : 'deleting';
    showLoading(`${deleteType} ${deletable.length} worksets...`);

    try {
        let successCount = 0;
        let errors = [];

        for (const worksetId of deletable) {
            try {
                await DaylilyAPI.worksets.delete(customerId, worksetId, hardDelete, reason || undefined);
                successCount++;
            } catch (e) {
                console.error(`Failed to delete ${worksetId}:`, e);
                errors.push(worksetId);
            }
        }

        if (successCount > 0) {
            const msg = hardDelete ? 'permanently deleted' : 'deleted';
            showToast('success', 'Bulk Delete Complete', `${successCount} of ${deletable.length} worksets ${msg}`);
        }
        if (errors.length > 0) {
            showToast('warning', 'Some Failed', `${errors.length} worksets failed to delete`);
        }

        setTimeout(() => window.location.reload(), 1500);
    } catch (error) {
        showToast('error', 'Bulk Delete Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Helper to get selected workset IDs
function getSelectedWorksets() {
    return Array.from(document.querySelectorAll('.workset-checkbox:checked')).map(cb => cb.value);
}

// Submit new workset
async function submitWorkset(event) {
    event.preventDefault();

    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) {
        showToast('error', 'Error', 'Customer ID not found');
        return;
    }

    const form = document.getElementById('workset-form');
    const formData = new FormData(form);

    const worksetName = formData.get('workset_name');

    // Generate a workset prefix for file uploads
    // This matches the server-side logic for workset ID generation
    const safeName = worksetName.replace(/\s+/g, '-').toLowerCase().substring(0, 30);
    const tempId = Math.random().toString(36).substring(2, 10);
    // Add date suffix in YYYYMMDD format
    const now = new Date();
    const dateSuffix = now.getFullYear().toString() +
        String(now.getMonth() + 1).padStart(2, '0') +
        String(now.getDate()).padStart(2, '0');
    const worksetPrefix = `worksets/${safeName}-${tempId}-${dateSuffix}/`;

    const data = {
        workset_name: worksetName,
        pipeline_type: formData.get('pipeline_type'),
        reference_genome: formData.get('reference_genome'),
        s3_bucket: formData.get('s3_bucket'),
        s3_prefix: worksetPrefix,  // Use the generated prefix
        priority: formData.get('priority'),
        workset_type: formData.get('workset_type') || 'ruo',
        notification_email: formData.get('notification_email'),
        enable_qc: formData.get('enable_qc') === 'on',
        archive_results: formData.get('archive_results') === 'on',
    };

    showLoading('Preparing workset...');

    try {
        // Upload selected files to S3 first (if any)
        const selectedFiles = window.getSelectedFiles ? window.getSelectedFiles() : [];
        if (selectedFiles.length > 0) {
            showLoading(`Uploading ${selectedFiles.length} file(s) to S3...`);
            try {
                const uploadResult = await window.uploadFilesToS3(customerId, worksetPrefix);
                if (!uploadResult.success) {
                    throw new Error('File upload failed');
                }
                showToast('success', 'Files Uploaded', `Uploaded ${uploadResult.uploadedFiles.length} file(s) to S3`);
            } catch (uploadError) {
                showToast('error', 'Upload Failed', uploadError.message);
                hideLoading();
                return;
            }
        }

        // Include samples from global worksetSamples array (now with S3 paths after upload)
        if (window.worksetSamples && window.worksetSamples.length > 0) {
            data.samples = window.worksetSamples;
        }

        // Include fileset if selected
        if (window.selectedFileset) {
            data.fileset_id = window.selectedFileset.fileset_id;
        }

        // Include manifest - either saved manifest ID or uploaded TSV content
        if (window.selectedManifestId) {
            data.manifest_id = window.selectedManifestId;
        } else if (window.manifestTsvContent) {
            data.manifest_tsv_content = window.manifestTsvContent;
        }

        // Include preferred cluster if selected
        const preferredCluster = document.getElementById('preferred_cluster')?.value;
        if (preferredCluster) {
            data.preferred_cluster = preferredCluster;
        }

        showLoading('Creating workset...');
        const result = await DaylilyAPI.worksets.create(customerId, data);
        showToast('success', 'Workset Submitted', 'Your workset has been queued for processing');
        setTimeout(() => {
            window.location.href = `/portal/worksets/${result.workset_id}`;
        }, 1500);
    } catch (error) {
        showToast('error', 'Submission Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Refresh workset detail page
async function refreshWorksetDetail(worksetState) {
    // For complete or error worksets, offer to re-gather stats from S3
    if (worksetState === 'complete' || worksetState === 'error') {
        const regatherStats = confirm(
            'Do you wish to re-gather stats from S3?\n\n' +
            'Click OK to re-fetch performance metrics from S3.\n' +
            'Click Cancel to just refresh the page.'
        );

        if (regatherStats) {
            // Re-gather stats by calling loadPerformanceMetrics with forceRefresh=true
            showLoading('Re-gathering stats from S3...');
            try {
                const customerId = window.UrsaConfig?.customerId || window.DaylilyConfig?.customerId;
                const worksetId = window.location.pathname.split('/').pop();

                const url = `/api/customers/${customerId}/worksets/${worksetId}/performance-metrics?force_refresh=true`;
                const response = await fetch(url);

                if (!response.ok) {
                    throw new Error('Failed to re-gather stats');
                }

                showToast('success', 'Stats Refreshed', 'Performance metrics have been re-gathered from S3');
                setTimeout(() => window.location.reload(), 1000);
            } catch (error) {
                showToast('error', 'Refresh Failed', error.message);
                hideLoading();
            }
            return;
        }
    }

    // Default: just reload the page
    window.location.reload();
}

// Download logs
async function downloadLogs(worksetId) {
    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;

    try {
        const logs = await DaylilyAPI.worksets.getLogs(customerId, worksetId);
        const blob = new Blob([logs.content], { type: 'text/plain' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `workset-${worksetId}-logs.txt`;
        a.click();
        URL.revokeObjectURL(url);
    } catch (error) {
        showToast('error', 'Download Failed', error.message);
    }
}

// Archive workset
async function archiveWorkset(worksetId) {
    const reason = prompt('Enter reason for archiving (optional):');
    if (reason === null) return; // User cancelled

    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;

    showLoading('Archiving workset...');

    try {
        await DaylilyAPI.worksets.archive(customerId, worksetId, reason || undefined);
        showToast('success', 'Workset Archived', 'The workset has been moved to archive');
        setTimeout(() => window.location.href = '/portal/worksets', 1500);
    } catch (error) {
        showToast('error', 'Archive Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Delete workset
async function deleteWorkset(worksetId) {
    const hardDelete = confirm('Do you want to permanently delete all data?\n\nClick OK for permanent deletion (cannot be undone)\nClick Cancel for soft delete (can be restored)');

    const confirmMsg = hardDelete
        ? 'Are you ABSOLUTELY sure you want to permanently delete this workset and ALL its data? This action CANNOT be undone!'
        : 'Delete this workset? It can be restored later if needed.';

    if (!confirm(confirmMsg)) return;

    const reason = prompt('Enter reason for deletion (optional):');
    if (reason === null) return; // User cancelled

    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;

    showLoading(hardDelete ? 'Permanently deleting workset...' : 'Deleting workset...');

    try {
        await DaylilyAPI.worksets.delete(customerId, worksetId, hardDelete, reason || undefined);
        showToast('success', 'Workset Deleted', hardDelete ? 'Workset permanently deleted' : 'Workset marked as deleted');
        setTimeout(() => window.location.href = '/portal/worksets', 1500);
    } catch (error) {
        showToast('error', 'Delete Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Restore archived workset
async function restoreWorkset(worksetId) {
    if (!confirm('Restore this workset? It will be set back to ready state.')) return;

    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;

    showLoading('Restoring workset...');

    try {
        await DaylilyAPI.worksets.restore(customerId, worksetId);
        showToast('success', 'Workset Restored', 'The workset has been restored');
        setTimeout(() => window.location.reload(), 1500);
    } catch (error) {
        showToast('error', 'Restore Failed', error.message);
    } finally {
        hideLoading();
    }
}

// Initialize checkbox listeners
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.workset-checkbox').forEach(cb => {
        cb.addEventListener('change', updateBulkActions);
    });

    // Initialize fileset selector if on new workset page
    if (document.getElementById('fileset-select')) {
        loadFilesetOptions();
    }

    // Initialize saved manifests dropdown if on new workset page
    if (document.getElementById('saved-manifest-select')) {
        refreshSavedManifestsList();
    }

    // Add cost calculation listeners
    const costTriggers = ['pipeline_type', 'reference_genome', 'priority', 'enable_qc', 'fileset-select'];
    costTriggers.forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.addEventListener('change', calculateCostEstimate);
        }
    });
});

// Tab switching for workset creation
function switchTab(tabName) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(tc => tc.classList.remove('active'));
    // Deactivate all tab items
    document.querySelectorAll('.tab-item').forEach(ti => ti.classList.remove('active'));

    // Show selected tab content
    const tabContent = document.getElementById(`tab-${tabName}`);
    if (tabContent) tabContent.classList.add('active');

    // Activate selected tab item
    const tabItems = document.querySelectorAll('.tab-item');
    tabItems.forEach(ti => {
        if (ti.textContent.toLowerCase().includes(tabName.toLowerCase()) ||
            ti.onclick?.toString().includes(`'${tabName}'`)) {
            ti.classList.add('active');
        }
    });

    // Store selected input method
    window.selectedInputMethod = tabName;

    // Recalculate cost
    calculateCostEstimate();
}

// Load fileset options for selector
async function loadFilesetOptions() {
    const select = document.getElementById('fileset-select');
    if (!select) return;

    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) return;

    try {
        const response = await fetch(`/api/v1/files/filesets?customer_id=${encodeURIComponent(customerId)}`);
        if (!response.ok) throw new Error('Failed to load filesets');

        const filesets = await response.json();

        select.innerHTML = '<option value="">Choose a file set...</option>';
        filesets.forEach(fs => {
            const option = document.createElement('option');
            option.value = fs.fileset_id;
            option.textContent = `${fs.name} (${fs.file_count} files)`;
            option.dataset.fileCount = fs.file_count;
            option.dataset.description = fs.description || '';
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load filesets:', error);
        select.innerHTML = '<option value="">Error loading file sets</option>';
    }
}

// Load fileset preview when selected
async function loadFilesetPreview() {
    const select = document.getElementById('fileset-select');
    const preview = document.getElementById('fileset-preview');
    if (!select || !preview) return;

    const filesetId = select.value;
    if (!filesetId) {
        preview.classList.add('d-none');
        window.selectedFileset = null;
        calculateCostEstimate();
        return;
    }

    try {
        const response = await fetch(`/api/v1/files/filesets/${filesetId}`);
        if (!response.ok) throw new Error('Failed to load fileset');

        const fileset = await response.json();
        window.selectedFileset = fileset;

        document.getElementById('fileset-name').textContent = fileset.name;
        document.getElementById('fileset-file-count').textContent = `${fileset.files?.length || 0} files`;
        document.getElementById('fileset-description').textContent = fileset.description || 'No description';

        // Show file preview (first 5 files)
        const filesPreview = document.getElementById('fileset-files-preview');
        if (filesPreview && fileset.files) {
            const displayFiles = fileset.files.slice(0, 5);
            filesPreview.innerHTML = displayFiles.map(f => `
                <div class="d-flex align-center gap-sm" style="padding: var(--spacing-xs) 0; border-bottom: 1px solid #2a3a44;">
                    <i class="fas fa-file text-muted"></i>
                    <span class="text-sm">${f.filename || f.file_id}</span>
                    <span class="text-muted text-sm ml-auto">${formatFileSize(f.file_size || 0)}</span>
                </div>
            `).join('');

            if (fileset.files.length > 5) {
                filesPreview.innerHTML += `<div class="text-muted text-sm mt-sm">...and ${fileset.files.length - 5} more files</div>`;
            }
        }

        preview.classList.remove('d-none');
        calculateCostEstimate();
    } catch (error) {
        console.error('Failed to load fileset preview:', error);
        preview.classList.add('d-none');
    }
}

// Preview manifest file
function previewManifest(input) {
    const file = input.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = function(e) {
        const content = e.target.result;
        const lines = content.split('\n').filter(l => l.trim());

        if (lines.length < 2) {
            showToast('error', 'Invalid Manifest', 'Manifest must have a header row and at least one data row');
            return;
        }

        const headers = lines[0].split('\t');
        const rows = lines.slice(1).map(line => line.split('\t'));

        // Build table
        const thead = document.getElementById('manifest-thead');
        const tbody = document.getElementById('manifest-tbody');

        thead.innerHTML = '<tr>' + headers.map(h => `<th>${h}</th>`).join('') + '</tr>';
        tbody.innerHTML = rows.slice(0, 10).map(row =>
            '<tr>' + row.map(cell => `<td>${cell}</td>`).join('') + '</tr>'
        ).join('');

        document.getElementById('manifest-row-count').textContent = rows.length;
        document.getElementById('manifest-preview').classList.remove('d-none');

        // Store raw TSV content for API submission (not parsed object)
        window.manifestTsvContent = content;
        window.manifestSampleCount = rows.length;
        // Clear saved manifest selection since user uploaded a new one
        window.selectedManifestId = null;
        const savedSelect = document.getElementById('saved-manifest-select');
        if (savedSelect) savedSelect.value = '';
        const savedPreview = document.getElementById('saved-manifest-preview');
        if (savedPreview) savedPreview.classList.add('d-none');

        calculateCostEstimate();
    };
    reader.readAsText(file);
}

// Refresh saved manifests list
async function refreshSavedManifestsList() {
    const customerId = window.UrsaConfig?.customerId;
    if (!customerId) {
        console.warn('No customer ID available for loading manifests');
        return;
    }

    const select = document.getElementById('saved-manifest-select');
    if (!select) return;

    try {
        const response = await DaylilyAPI.manifests.list(customerId);
        const manifests = response.manifests || [];

        // Clear existing options except the placeholder
        select.innerHTML = '<option value="">Choose a saved manifest...</option>';

        if (manifests.length === 0) {
            select.innerHTML += '<option value="" disabled>No saved manifests found</option>';
            return;
        }

        manifests.forEach(m => {
            const option = document.createElement('option');
            option.value = m.manifest_id;
            const name = m.name || m.manifest_id;
            // Ensure sample_count is a number (API returns int, but be defensive)
            const sampleCount = typeof m.sample_count === 'number' ? m.sample_count : parseInt(m.sample_count, 10) || 0;
            const created = m.created_at ? new Date(m.created_at).toLocaleDateString() : '';
            option.textContent = `${name} (${sampleCount} samples${created ? ', ' + created : ''})`;
            option.dataset.name = name;
            option.dataset.sampleCount = String(sampleCount);  // Store as string for dataset
            option.dataset.createdAt = m.created_at || '';
            select.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load saved manifests:', error);
        showToast('error', 'Error', 'Failed to load saved manifests');
    }
}

// Handle saved manifest selection
function onSavedManifestSelected() {
    const select = document.getElementById('saved-manifest-select');
    const manifestId = select?.value;
    const preview = document.getElementById('saved-manifest-preview');

    if (!manifestId) {
        // Clear selection state
        if (preview) preview.classList.add('d-none');
        window.selectedManifestId = null;
        window.manifestSampleCount = 0;
        calculateCostEstimate();
        return;
    }

    try {
        // Get manifest metadata from the selected option's dataset
        const option = select.options[select.selectedIndex];
        const name = option.dataset.name || manifestId;
        // Parse sample count - dataset attributes are always strings
        const sampleCountStr = option.dataset.sampleCount;
        const sampleCount = sampleCountStr ? parseInt(sampleCountStr, 10) : 0;
        const createdAt = option.dataset.createdAt;

        console.log('Selected manifest:', manifestId, 'sampleCount:', sampleCount, 'raw:', sampleCountStr);

        // Update preview
        const nameEl = document.getElementById('saved-manifest-name');
        const countEl = document.getElementById('saved-manifest-sample-count');
        const createdEl = document.getElementById('saved-manifest-created');

        if (nameEl) nameEl.textContent = name;
        if (countEl) countEl.textContent = `${sampleCount} samples`;
        if (createdEl && createdAt) {
            createdEl.textContent = `Created: ${new Date(createdAt).toLocaleString()}`;
        }

        if (preview) preview.classList.remove('d-none');

        // Store selected manifest ID for submission
        window.selectedManifestId = manifestId;
        window.manifestSampleCount = sampleCount;
        // Clear uploaded manifest since user selected a saved one
        window.manifestTsvContent = null;
        const uploadPreview = document.getElementById('manifest-preview');
        if (uploadPreview) uploadPreview.classList.add('d-none');
        const manifestInput = document.getElementById('manifest-input');
        if (manifestInput) manifestInput.value = '';

        // Trigger cost calculation
        calculateCostEstimate();
    } catch (error) {
        console.error('Failed to load manifest details:', error);
        showToast('error', 'Error', 'Failed to load manifest details');
    }
}

// Calculate cost estimate
function calculateCostEstimate() {
    const pipeline = document.getElementById('pipeline_type')?.value;
    const reference = document.getElementById('reference_genome')?.value;
    const priority = document.getElementById('priority')?.value || 'normal';
    const enableQc = document.getElementById('enable_qc')?.checked ?? true;

    console.log('calculateCostEstimate called:', {
        pipeline,
        priority,
        enableQc,
        selectedFileset: !!window.selectedFileset,
        manifestSampleCount: window.manifestSampleCount,
        selectedManifestId: window.selectedManifestId
    });

    // Determine sample count based on input method
    let sampleCount = 0;
    let totalDataSize = 0; // in GB

    if (window.selectedFileset) {
        sampleCount = Math.ceil((window.selectedFileset.files?.length || 0) / 2); // Assume paired-end
        totalDataSize = (window.selectedFileset.files || []).reduce((sum, f) => sum + (f.file_size || 0), 0) / (1024 * 1024 * 1024);
        console.log('Using selectedFileset:', sampleCount, 'samples');
    } else if (window.manifestSampleCount > 0) {
        // Use sample count from uploaded or selected manifest
        sampleCount = window.manifestSampleCount;
        totalDataSize = sampleCount * 30; // Estimate 30GB per sample
        console.log('Using manifestSampleCount:', sampleCount, 'samples');
    } else if (window.selectedManifestId && !window.manifestSampleCount) {
        // Fallback: manifest selected but sample count not available (legacy data)
        // Estimate 1 sample as minimum to show some cost estimate
        sampleCount = 1;
        totalDataSize = 30;
        console.warn('Manifest selected but sample_count not available, using estimate');
    } else if (window.worksetSamples?.length) {
        sampleCount = window.worksetSamples.length;
        totalDataSize = sampleCount * 30;
        console.log('Using worksetSamples:', sampleCount, 'samples');
    } else {
        console.log('No sample source found, sampleCount=0');
    }

    // Base costs per sample (in USD) - map actual pipeline types to cost profiles
    const pipelineCosts = {
        // Test pipeline
        'test_help': { compute: 0.50, vcpuHours: 2, timeHours: 0.5 },
        // Germline WGS variants
        'germline_wgs_snv': { compute: 2.50, vcpuHours: 8, timeHours: 2 },
        'germline_wgs_snv_sv': { compute: 3.50, vcpuHours: 12, timeHours: 3 },
        'germline_wgs_kitchensink': { compute: 5.00, vcpuHours: 16, timeHours: 4 },
        // Legacy/fallback keys
        'germline': { compute: 2.50, vcpuHours: 8, timeHours: 2 },
        'somatic': { compute: 5.00, vcpuHours: 16, timeHours: 4 },
        'rnaseq': { compute: 1.50, vcpuHours: 4, timeHours: 1.5 },
        'wgs': { compute: 8.00, vcpuHours: 32, timeHours: 6 },
        'wes': { compute: 3.00, vcpuHours: 12, timeHours: 2.5 }
    };

    // Default cost profile for when pipeline not yet selected
    const defaultCost = { compute: 2.50, vcpuHours: 8, timeHours: 2 };

    // Get cost profile - use default if pipeline not selected or unknown
    const baseCost = (pipeline && pipelineCosts[pipeline]) ? pipelineCosts[pipeline] : defaultCost;

    // If no samples, show zeros
    if (sampleCount === 0) {
        updateCostDisplay(0, 0, 0, 0, 0, 0, 0, 1.0);
        return;
    }

    // Priority multipliers
    const priorityMultipliers = {
        'low': 0.5,
        'normal': 1.0,
        'high': 2.0
    };
    const priorityMult = priorityMultipliers[priority] || 1.0;

    // Calculate costs
    let computeCost = baseCost.compute * sampleCount;
    let storageCost = totalDataSize * 0.023; // S3 standard pricing per GB/month
    let transferCost = totalDataSize * 0.09; // Data transfer out

    // QC adds 5% to compute
    if (enableQc) {
        computeCost *= 1.05;
    }

    // Apply priority multiplier to compute only
    computeCost *= priorityMult;

    const totalCost = computeCost + storageCost + transferCost;
    const totalTime = baseCost.timeHours * sampleCount / 4; // Parallel processing
    const totalVcpu = baseCost.vcpuHours * sampleCount;

    updateCostDisplay(totalCost, totalTime, totalVcpu, sampleCount, computeCost, storageCost, transferCost, priorityMult);
}

// Update cost display
function updateCostDisplay(total, time, vcpu, samples, compute, storage, transfer, priorityMult) {
    console.log('updateCostDisplay:', { total, time, vcpu, samples, compute, storage, transfer, priorityMult });
    document.getElementById('est-cost').textContent = `$${total.toFixed(2)}`;
    document.getElementById('est-time').textContent = time > 0 ? `${time.toFixed(1)}h` : '0h';
    document.getElementById('est-vcpu').textContent = vcpu.toFixed(0);
    document.getElementById('est-samples').textContent = samples;

    // Update breakdown if elements exist
    const computeEl = document.getElementById('cost-compute');
    const storageEl = document.getElementById('cost-storage');
    const transferEl = document.getElementById('cost-transfer');
    const priorityEl = document.getElementById('cost-priority');

    if (computeEl) computeEl.textContent = `$${compute.toFixed(2)}`;
    if (storageEl) storageEl.textContent = `$${storage.toFixed(2)}`;
    if (transferEl) transferEl.textContent = `$${transfer.toFixed(2)}`;
    if (priorityEl) priorityEl.textContent = `${priorityMult}x`;
}

// Format file size helper
function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ========== Cluster Selection Functions ==========

// Store loaded clusters for suggestion logic
let availableClusters = [];

/**
 * Refresh the cluster list from the API and populate the dropdown.
 */
async function refreshClusterList() {
    const select = document.getElementById('preferred_cluster');
    const tbody = document.getElementById('cluster-list-body');
    const container = document.getElementById('cluster-list-container');
    const suggestionText = document.getElementById('cluster-suggestion-text');

    if (!select) return;

    // Show loading state
    select.innerHTML = '<option value="">Loading clusters...</option>';
    if (tbody) {
        tbody.innerHTML = '<tr><td colspan="5" class="text-center"><i class="fas fa-spinner fa-spin"></i> Loading...</td></tr>';
    }
    if (container) container.style.display = 'block';

    try {
        // Use cached cluster data - refresh is expensive (10+ seconds)
        // The cache has 5-minute TTL and parallel region scanning
        const response = await fetch('/api/clusters');
        if (!response.ok) throw new Error(`Failed to load clusters (${response.status})`);

        const data = await response.json();
        availableClusters = data.clusters || [];

        // Populate the dropdown
        select.innerHTML = '<option value="">Auto-select (recommended)</option>';
        let tableHtml = '';

        // Get detected file regions from manifest generator (if available)
        const r1Region = window.r1FileRegion;
        const r2Region = window.r2FileRegion;
        const fileRegion = r1Region || r2Region;

        // Sort clusters: running first, then by region match
        const sortedClusters = [...availableClusters].sort((a, b) => {
            // Prefer running clusters
            const aRunning = a.cluster_status === 'CREATE_COMPLETE';
            const bRunning = b.cluster_status === 'CREATE_COMPLETE';
            if (aRunning && !bRunning) return -1;
            if (!aRunning && bRunning) return 1;

            // Then prefer region match
            if (fileRegion) {
                const aMatch = a.region === fileRegion;
                const bMatch = b.region === fileRegion;
                if (aMatch && !bMatch) return -1;
                if (!aMatch && bMatch) return 1;
            }

            return a.cluster_name.localeCompare(b.cluster_name);
        });

        sortedClusters.forEach(cluster => {
            const isRunning = cluster.cluster_status === 'CREATE_COMPLETE';
            const regionMatch = fileRegion && cluster.region === fileRegion;
            const statusClass = isRunning ? 'badge-success' : 'badge-secondary';
            const statusText = isRunning ? 'Running' : (cluster.cluster_status || 'Unknown');
            const matchBadge = regionMatch
                ? '<span class="badge badge-info" title="Same region as your data"><i class="fas fa-check"></i> Match</span>'
                : '<span class="text-muted">—</span>';

            // Add to dropdown
            const optionText = `${cluster.cluster_name} (${cluster.region})${regionMatch ? ' ★' : ''}`;
            select.innerHTML += `<option value="${cluster.cluster_name}" data-region="${cluster.region}">${optionText}</option>`;

            // Add to table
            tableHtml += `
                <tr onclick="selectCluster('${cluster.cluster_name}')" style="cursor: pointer;">
                    <td><input type="radio" name="cluster_radio" value="${cluster.cluster_name}"></td>
                    <td><code style="font-size: 0.8rem;">${cluster.cluster_name}</code></td>
                    <td><span class="badge badge-secondary" style="font-size: 0.7rem;">${cluster.region}</span></td>
                    <td><span class="badge ${statusClass}" style="font-size: 0.7rem;">${statusText}</span></td>
                    <td>${matchBadge}</td>
                </tr>
            `;
        });

        if (tbody) tbody.innerHTML = tableHtml || '<tr><td colspan="5" class="text-center text-muted">No clusters available</td></tr>';

        // Update suggestion text
        if (suggestionText && fileRegion) {
            const matchingClusters = availableClusters.filter(c => c.region === fileRegion && c.cluster_status === 'CREATE_COMPLETE');
            if (matchingClusters.length > 0) {
                suggestionText.innerHTML = `<i class="fas fa-lightbulb text-warning"></i> <strong>Recommended:</strong> ${matchingClusters.length} cluster(s) in <code>${fileRegion}</code> match your data location.`;
            } else {
                suggestionText.innerHTML = `<i class="fas fa-exclamation-triangle text-warning"></i> No running clusters in <code>${fileRegion}</code>. Data transfer costs may apply.`;
            }
        }

    } catch (error) {
        console.error('Failed to load clusters:', error);
        select.innerHTML = '<option value="">Auto-select (recommended)</option>';
        if (tbody) {
            tbody.innerHTML = `<tr><td colspan="5" class="text-center text-error"><i class="fas fa-exclamation-triangle"></i> Failed to load clusters</td></tr>`;
        }
    }
}

/**
 * Select a cluster from the table view.
 */
function selectCluster(clusterName) {
    const select = document.getElementById('preferred_cluster');
    if (select) {
        select.value = clusterName;
    }
    // Update radio button
    const radio = document.querySelector(`input[name="cluster_radio"][value="${clusterName}"]`);
    if (radio) radio.checked = true;
}

// Load clusters when the page loads (if on workset new page)
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('preferred_cluster')) {
        refreshClusterList();
    }
});

