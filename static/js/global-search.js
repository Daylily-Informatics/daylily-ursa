(function () {
    'use strict';

    var STORAGE_KEY = 'ursa.global_search.query';

    function isTypingElement(element) {
        if (!element) {
            return false;
        }
        var tag = (element.tagName || '').toUpperCase();
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') {
            return true;
        }
        return Boolean(element.isContentEditable);
    }

    function syncFacetHidden(facetsForm, fieldName, fieldValue) {
        if (!facetsForm) {
            return;
        }
        var hidden = facetsForm.querySelector('input[name="' + fieldName + '"]');
        if (hidden) {
            hidden.value = fieldValue;
        }
    }

    function initGlobalSearch() {
        var headerInput = document.getElementById('global-search-input');
        var queryForms = document.querySelectorAll('[data-global-search-form]');
        var facetsForm = document.getElementById('portal-search-facets');
        var pageSearchForm = document.getElementById('portal-search-form');

        queryForms.forEach(function (form) {
            form.addEventListener('submit', function () {
                var input = form.querySelector('input[name="q"]');
                if (!input) {
                    return;
                }
                var value = (input.value || '').trim();
                if (value) {
                    sessionStorage.setItem(STORAGE_KEY, value);
                } else {
                    sessionStorage.removeItem(STORAGE_KEY);
                }
            });
        });

        if (headerInput && !headerInput.value.trim()) {
            var saved = sessionStorage.getItem(STORAGE_KEY) || '';
            if (saved) {
                headerInput.value = saved;
            }
        }

        document.addEventListener('keydown', function (event) {
            if (event.defaultPrevented || event.key !== '/') {
                return;
            }
            if (event.ctrlKey || event.metaKey || event.altKey) {
                return;
            }
            if (isTypingElement(document.activeElement)) {
                return;
            }
            if (!headerInput) {
                return;
            }
            event.preventDefault();
            headerInput.focus();
            headerInput.select();
        });

        if (pageSearchForm && facetsForm) {
            var queryInput = pageSearchForm.querySelector('input[name="q"]');
            if (queryInput) {
                queryInput.addEventListener('input', function () {
                    syncFacetHidden(facetsForm, 'q', queryInput.value || '');
                });
            }

            var scopeSelect = pageSearchForm.querySelector('select[name="scope"]');
            if (scopeSelect) {
                scopeSelect.addEventListener('change', function () {
                    syncFacetHidden(facetsForm, 'scope', scopeSelect.value || 'mine');
                });
            }

            facetsForm.querySelectorAll('[data-submit-on-change]').forEach(function (element) {
                element.addEventListener('change', function () {
                    facetsForm.submit();
                });
            });
        }
    }

    document.addEventListener('DOMContentLoaded', initGlobalSearch);
})();
