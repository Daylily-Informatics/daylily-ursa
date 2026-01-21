/**
 * Table Utilities - Sorting and TSV Download
 * Auto-initializes on DOMContentLoaded for tables with data-sortable attribute
 */

(function() {
    'use strict';

    // Track sort state per table
    const sortState = new Map();

    /**
     * Initialize sortable tables
     */
    function initSortableTables() {
        document.querySelectorAll('table[data-sortable]').forEach(table => {
            initTableSort(table);
            addDownloadButton(table);
        });
    }

    /**
     * Initialize sorting for a single table
     */
    function initTableSort(table) {
        const headers = table.querySelectorAll('thead th');
        const tableId = table.id || `table-${Math.random().toString(36).substr(2, 9)}`;
        table.id = tableId;
        
        sortState.set(tableId, { column: null, direction: 'asc' });

        headers.forEach((th, index) => {
            // Skip columns marked as non-sortable (checkboxes, actions, etc.)
            if (th.hasAttribute('data-no-sort') || 
                th.querySelector('input[type="checkbox"]') ||
                th.textContent.trim().toLowerCase() === 'actions') {
                return;
            }

            th.classList.add('sortable');
            th.setAttribute('data-col-index', index);
            th.style.cursor = 'pointer';
            th.style.userSelect = 'none';
            
            // Add sort indicator
            const indicator = document.createElement('span');
            indicator.className = 'sort-indicator';
            indicator.innerHTML = ' <i class="fas fa-sort"></i>';
            th.appendChild(indicator);

            th.addEventListener('click', () => sortTable(table, index, th));
        });
    }

    /**
     * Sort table by column
     */
    function sortTable(table, colIndex, header) {
        const tableId = table.id;
        const state = sortState.get(tableId);
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        // Determine sort direction
        let direction = 'asc';
        if (state.column === colIndex) {
            direction = state.direction === 'asc' ? 'desc' : 'asc';
        }
        
        // Update state
        state.column = colIndex;
        state.direction = direction;
        
        // Update header indicators
        table.querySelectorAll('thead th .sort-indicator').forEach(ind => {
            ind.innerHTML = ' <i class="fas fa-sort"></i>';
        });
        const indicator = header.querySelector('.sort-indicator');
        if (indicator) {
            indicator.innerHTML = direction === 'asc' 
                ? ' <i class="fas fa-sort-up"></i>' 
                : ' <i class="fas fa-sort-down"></i>';
        }
        
        // Sort rows
        rows.sort((a, b) => {
            const aCell = a.cells[colIndex];
            const bCell = b.cells[colIndex];
            if (!aCell || !bCell) return 0;

            let aVal = getCellSortValue(aCell);
            let bVal = getCellSortValue(bCell);

            // Try date comparison first (check for ISO-like patterns)
            const datePattern = /^\d{4}-\d{2}-\d{2}/;
            if (datePattern.test(aVal) && datePattern.test(bVal)) {
                const aDate = Date.parse(aVal);
                const bDate = Date.parse(bVal);
                if (!isNaN(aDate) && !isNaN(bDate)) {
                    return direction === 'asc' ? aDate - bDate : bDate - aDate;
                }
            }

            // Try numeric comparison
            const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
            const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));

            if (!isNaN(aNum) && !isNaN(bNum)) {
                return direction === 'asc' ? aNum - bNum : bNum - aNum;
            }

            // Fall back to string comparison
            return direction === 'asc'
                ? aVal.localeCompare(bVal)
                : bVal.localeCompare(aVal);
        });
        
        // Re-append sorted rows
        rows.forEach(row => tbody.appendChild(row));
    }

    /**
     * Get sortable value from cell
     */
    function getCellSortValue(cell) {
        // Check for data-sort-value attribute
        if (cell.hasAttribute('data-sort-value')) {
            return cell.getAttribute('data-sort-value');
        }
        // Use text content, clean up whitespace
        return cell.textContent.trim().toLowerCase();
    }

    /**
     * Add download button above table
     */
    function addDownloadButton(table) {
        const tableId = table.id;
        const container = table.closest('.table-container') || table.parentElement;
        
        // Check if button already exists
        if (container.querySelector('.table-download-btn')) return;
        
        // Create button container
        const btnContainer = document.createElement('div');
        btnContainer.className = 'table-actions d-flex justify-end mb-sm';
        btnContainer.innerHTML = `
            <button class="btn btn-outline btn-sm table-download-btn" onclick="downloadTableAsTSV('${tableId}')">
                <i class="fas fa-download"></i> Download TSV
            </button>
        `;
        
        container.insertBefore(btnContainer, table);
    }

    /**
     * Download table as TSV
     */
    window.downloadTableAsTSV = function(tableId) {
        const table = document.getElementById(tableId);
        if (!table) {
            console.error('Table not found:', tableId);
            return;
        }
        
        const rows = [];
        const headers = [];
        
        // Get headers (skip checkbox and actions columns)
        table.querySelectorAll('thead th').forEach((th, i) => {
            if (!th.hasAttribute('data-no-sort') && 
                !th.querySelector('input[type="checkbox"]') &&
                th.textContent.trim().toLowerCase() !== 'actions') {
                headers.push({ index: i, text: th.textContent.replace(/[\n\r]/g, ' ').trim().split(' ')[0] });
            }
        });
        
        rows.push(headers.map(h => h.text).join('\t'));
        
        // Get visible rows only
        table.querySelectorAll('tbody tr').forEach(tr => {
            if (tr.style.display === 'none') return;
            
            const rowData = [];
            headers.forEach(h => {
                const cell = tr.cells[h.index];
                if (cell) {
                    // Get clean text value
                    let val = cell.getAttribute('data-sort-value') || cell.textContent;
                    val = val.replace(/[\n\r\t]/g, ' ').trim();
                    rowData.push(val);
                }
            });
            if (rowData.length > 0) {
                rows.push(rowData.join('\t'));
            }
        });
        
        // Create and trigger download
        const tsv = rows.join('\n');
        const blob = new Blob([tsv], { type: 'text/tab-separated-values' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${tableId}-${new Date().toISOString().split('T')[0]}.tsv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        if (typeof showToast === 'function') {
            showToast('success', 'Downloaded', `Table exported as TSV`);
        }
    };

    // Initialize on DOM ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSortableTables);
    } else {
        initSortableTables();
    }

    // Expose for manual initialization
    window.initSortableTables = initSortableTables;
})();

