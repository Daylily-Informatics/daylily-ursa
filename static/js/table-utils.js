/**
 * Table Utilities - Sorting, Filtering, and TSV Download
 * Auto-initializes on all tables unless explicitly disabled via data-table-tools="off".
 */

(function() {
    'use strict';

    const sortState = new Map();
    let generatedTableCounter = 0;
    let tableObserver = null;

    function shouldEnhanceTable(table) {
        return table && table.tagName === 'TABLE' && table.getAttribute('data-table-tools') !== 'off';
    }

    function ensureTableId(table) {
        if (table.id && table.id.trim()) {
            return table.id;
        }
        generatedTableCounter += 1;
        table.id = `table-tools-${generatedTableCounter}`;
        return table.id;
    }

    function getTableHeaders(table) {
        return Array.from(table.querySelectorAll('thead th'));
    }

    function getTableRows(table) {
        const tbody = table.querySelector('tbody');
        if (!tbody) {
            return [];
        }
        return Array.from(tbody.querySelectorAll('tr'));
    }

    function getHeaderText(th) {
        const clone = th.cloneNode(true);
        clone.querySelectorAll('.sort-indicator').forEach(node => node.remove());
        return clone.textContent.replace(/\s+/g, ' ').trim();
    }

    function isNoSortHeader(th) {
        const label = getHeaderText(th).toLowerCase();
        return (
            th.hasAttribute('data-no-sort') ||
            th.querySelector('input[type="checkbox"]') ||
            label === '' ||
            label === 'actions'
        );
    }

    function initTableSort(table) {
        const headers = getTableHeaders(table);
        if (headers.length === 0) return;

        const tableId = ensureTableId(table);
        if (!sortState.has(tableId)) {
            sortState.set(tableId, { column: null, direction: 'asc' });
        }

        headers.forEach((th, index) => {
            if (isNoSortHeader(th)) {
                return;
            }

            th.classList.add('sortable');
            th.setAttribute('data-col-index', index);

            if (!th.querySelector('.sort-indicator')) {
                const indicator = document.createElement('span');
                indicator.className = 'sort-indicator';
                indicator.innerHTML = ' <i class="fas fa-sort"></i>';
                th.appendChild(indicator);
            }

            if (!th.dataset.tableSortBound) {
                th.addEventListener('click', () => sortTable(table, index, th));
                th.dataset.tableSortBound = '1';
            }
        });
    }

    function getCellSortValue(cell) {
        if (cell.hasAttribute('data-sort-value')) {
            return String(cell.getAttribute('data-sort-value') || '').trim().toLowerCase();
        }
        return cell.textContent.replace(/\s+/g, ' ').trim().toLowerCase();
    }

    function sortTable(table, colIndex, header) {
        const tableId = ensureTableId(table);
        const state = sortState.get(tableId) || { column: null, direction: 'asc' };
        const tbody = table.querySelector('tbody');
        if (!tbody) return;

        const rows = Array.from(tbody.querySelectorAll('tr'));
        if (!rows.length) return;

        let direction = 'asc';
        if (state.column === colIndex) {
            direction = state.direction === 'asc' ? 'desc' : 'asc';
        }
        state.column = colIndex;
        state.direction = direction;
        sortState.set(tableId, state);

        table.querySelectorAll('thead th .sort-indicator').forEach(ind => {
            ind.innerHTML = ' <i class="fas fa-sort"></i>';
        });

        const indicator = header.querySelector('.sort-indicator');
        if (indicator) {
            indicator.innerHTML = direction === 'asc'
                ? ' <i class="fas fa-sort-up"></i>'
                : ' <i class="fas fa-sort-down"></i>';
        }

        rows.sort((a, b) => {
            const aCell = a.cells[colIndex];
            const bCell = b.cells[colIndex];
            if (!aCell || !bCell) return 0;

            const aVal = getCellSortValue(aCell);
            const bVal = getCellSortValue(bCell);

            const aDate = Date.parse(aVal);
            const bDate = Date.parse(bVal);
            if (!Number.isNaN(aDate) && !Number.isNaN(bDate)) {
                return direction === 'asc' ? aDate - bDate : bDate - aDate;
            }

            const aNum = parseFloat(aVal.replace(/[^0-9.-]/g, ''));
            const bNum = parseFloat(bVal.replace(/[^0-9.-]/g, ''));
            if (!Number.isNaN(aNum) && !Number.isNaN(bNum)) {
                return direction === 'asc' ? aNum - bNum : bNum - aNum;
            }

            return direction === 'asc'
                ? aVal.localeCompare(bVal)
                : bVal.localeCompare(aVal);
        });

        rows.forEach(row => tbody.appendChild(row));

        const filterInput = table.closest('.table-tools-shell')?.querySelector('.table-filter-input');
        if (filterInput) {
            filterTableRows(table, filterInput.value || '');
        }
    }

    function filterTableRows(table, query) {
        const normalizedQuery = String(query || '').trim().toLowerCase();
        getTableRows(table).forEach(row => {
            if (!row.dataset.tableOriginalDisplay) {
                row.dataset.tableOriginalDisplay = row.style.display || '';
            }

            if (!normalizedQuery) {
                row.style.display = row.dataset.tableOriginalDisplay;
                return;
            }

            const text = row.textContent.replace(/\s+/g, ' ').trim().toLowerCase();
            row.style.display = text.includes(normalizedQuery) ? '' : 'none';
        });
    }

    function getExportColumns(table) {
        const columns = [];
        getTableHeaders(table).forEach((th, index) => {
            if (isNoSortHeader(th)) {
                return;
            }
            const header = getHeaderText(th);
            columns.push({
                index,
                text: header || `column_${index + 1}`,
            });
        });
        return columns;
    }

    function createToolsShell(table) {
        const tableId = ensureTableId(table);
        const host = table.closest('.table-container, .table-wrapper') || table.parentElement;
        if (!host) return null;

        let shell = host.querySelector(`.table-tools-shell[data-table-id="${tableId}"]`);
        if (shell) return shell;

        shell = document.createElement('div');
        shell.className = 'table-actions table-tools-shell d-flex justify-between align-center flex-wrap gap-sm mb-sm';
        shell.setAttribute('data-table-id', tableId);
        shell.innerHTML = `
            <div class="table-filter-wrap">
                <input type="text" class="form-control table-filter-input" placeholder="Filter table..." aria-label="Filter table rows">
            </div>
            <div class="table-download-wrap">
                <button type="button" class="btn btn-outline btn-sm table-download-btn">
                    <i class="fas fa-download"></i> Download TSV
                </button>
            </div>
        `;

        const filterInput = shell.querySelector('.table-filter-input');
        const downloadButton = shell.querySelector('.table-download-btn');

        if (filterInput) {
            filterInput.addEventListener('input', () => filterTableRows(table, filterInput.value || ''));
        }
        if (downloadButton) {
            downloadButton.addEventListener('click', () => downloadTableAsTSV(tableId));
        }

        host.insertBefore(shell, table);
        return shell;
    }

    function initSingleTable(table) {
        if (!shouldEnhanceTable(table)) return;
        if (table.dataset.tableToolsInitialized === '1') return;

        ensureTableId(table);
        initTableSort(table);
        createToolsShell(table);
        table.dataset.tableToolsInitialized = '1';
    }

    function initSortableTables(root) {
        const scope = root || document;
        const tables = [];

        if (scope instanceof Element && scope.tagName === 'TABLE') {
            tables.push(scope);
        }
        if (scope.querySelectorAll) {
            scope.querySelectorAll('table').forEach(table => tables.push(table));
        }

        const uniqueTables = Array.from(new Set(tables));
        uniqueTables.forEach(initSingleTable);
    }

    function getVisibleRows(table) {
        return getTableRows(table).filter(row => row.style.display !== 'none');
    }

    window.downloadTableAsTSV = function(tableId) {
        const table = document.getElementById(tableId);
        if (!table) {
            console.error('Table not found:', tableId);
            return;
        }

        const columns = getExportColumns(table);
        if (!columns.length) {
            console.warn('No exportable columns found:', tableId);
            return;
        }

        const lines = [];
        lines.push(columns.map(c => c.text).join('\t'));

        getVisibleRows(table).forEach(row => {
            const values = columns.map(col => {
                const cell = row.cells[col.index];
                if (!cell) return '';
                const val = cell.getAttribute('data-sort-value') || cell.textContent;
                return String(val).replace(/[\n\r\t]/g, ' ').trim();
            });
            lines.push(values.join('\t'));
        });

        const tsv = lines.join('\n');
        const blob = new Blob([tsv], { type: 'text/tab-separated-values' });
        const url = URL.createObjectURL(blob);
        const anchor = document.createElement('a');
        anchor.href = url;
        anchor.download = `${tableId}-${new Date().toISOString().split('T')[0]}.tsv`;
        document.body.appendChild(anchor);
        anchor.click();
        document.body.removeChild(anchor);
        URL.revokeObjectURL(url);

        if (typeof showToast === 'function') {
            showToast('success', 'Downloaded', 'Table exported as TSV');
        }
    };

    function observeDynamicTables() {
        if (tableObserver || !document.body) return;
        tableObserver = new MutationObserver(mutations => {
            mutations.forEach(mutation => {
                mutation.addedNodes.forEach(node => {
                    if (!(node instanceof Element)) return;
                    if (node.tagName === 'TABLE') {
                        initSortableTables(node);
                    } else if (node.querySelectorAll) {
                        initSortableTables(node);
                    }
                });
            });
        });
        tableObserver.observe(document.body, { childList: true, subtree: true });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            initSortableTables(document);
            observeDynamicTables();
        });
    } else {
        initSortableTables(document);
        observeDynamicTables();
    }

    window.initSortableTables = initSortableTables;
})();
