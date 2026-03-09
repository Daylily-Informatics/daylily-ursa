(function () {
    const DEFAULT_TZ = 'UTC';

    function normalizeTimezone(value) {
        const candidate = String(value || '').trim();
        if (!candidate) return DEFAULT_TZ;
        const upper = candidate.toUpperCase();
        if (['UTC', 'GMT', 'GMT+00:00', 'Z'].includes(upper)) return DEFAULT_TZ;
        try {
            new Intl.DateTimeFormat('en-US', { timeZone: candidate });
            return candidate;
        } catch (_err) {
            return DEFAULT_TZ;
        }
    }

    function getDisplayTimezone() {
        return normalizeTimezone(window.UrsaConfig?.displayTimezone || DEFAULT_TZ);
    }

    function parseDate(value) {
        if (!value) return null;
        if (value instanceof Date) return value;
        const parsed = new Date(value);
        if (Number.isNaN(parsed.getTime())) return null;
        return parsed;
    }

    function formatDate(value, options = {}) {
        const parsed = parseDate(value);
        if (!parsed) return '';
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            timeZone: getDisplayTimezone(),
            ...options,
        }).format(parsed);
    }

    function formatDateTime(value, options = {}) {
        const parsed = parseDate(value);
        if (!parsed) return '';
        return new Intl.DateTimeFormat('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            timeZone: getDisplayTimezone(),
            ...options,
        }).format(parsed);
    }

    function formatMonthDay(value) {
        const parsed = parseDate(value);
        if (!parsed) return '';
        return new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
            timeZone: getDisplayTimezone(),
        }).format(parsed);
    }

    window.UrsaTime = {
        normalizeTimezone,
        getDisplayTimezone,
        formatDate,
        formatDateTime,
        formatMonthDay,
    };
})();
