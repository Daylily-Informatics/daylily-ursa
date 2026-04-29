window.UrsaPortal = (() => {
  function initMenu() {
    const toggle = document.querySelector("[data-menu-toggle]");
    const links = document.getElementById("nav-links");
    if (!toggle || !links) {
      return;
    }
    toggle.addEventListener("click", () => {
      links.classList.toggle("active");
    });
  }

  function initUserMenu() {
    const avatar = document.getElementById("user-avatar");
    const dropdown = document.getElementById("user-dropdown");
    if (!avatar || !dropdown) {
      return;
    }
    avatar.addEventListener("click", (event) => {
      event.stopPropagation();
      dropdown.classList.toggle("active");
    });
    document.addEventListener("click", () => dropdown.classList.remove("active"));
  }

  function parseToastArgs(arg1, arg2, arg3) {
    if (arg3 !== undefined) {
      return { type: String(arg1 || "info"), title: String(arg2 || ""), message: String(arg3 || "") };
    }
    if (arg2 !== undefined) {
      const normalized = String(arg2 || "").toLowerCase();
      if (["success", "error", "info", "warning"].includes(normalized)) {
        return { type: normalized, title: "", message: String(arg1 || "") };
      }
      return { type: String(arg1 || "info"), title: "", message: String(arg2 || "") };
    }
    return { type: "info", title: "", message: String(arg1 || "") };
  }

  function showToast(arg1, arg2, arg3) {
    const { type, title, message } = parseToastArgs(arg1, arg2, arg3);
    const container = document.getElementById("toast-container");
    if (!container) {
      return;
    }
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.innerHTML = title
      ? `<strong>${escapeHtml(title)}</strong><div>${escapeHtml(message)}</div>`
      : escapeHtml(message);
    container.appendChild(toast);
    window.setTimeout(() => toast.remove(), 5000);
  }

  function showLoading(message = "Loading...") {
    const overlay = document.getElementById("loading-overlay");
    if (!overlay) {
      return;
    }
    const label = overlay.querySelector("p");
    if (label) {
      label.textContent = message;
    }
    overlay.classList.remove("d-none");
  }

  function hideLoading() {
    const overlay = document.getElementById("loading-overlay");
    if (overlay) {
      overlay.classList.add("d-none");
    }
  }

  function showModal(id) {
    const modal = document.getElementById(id);
    if (!modal) {
      return;
    }
    modal.classList.add("active");
  }

  function closeModal(id) {
    const modal = document.getElementById(id);
    if (!modal) {
      return;
    }
    modal.classList.remove("active");
  }

  function escapeHtml(value) {
    const node = document.createElement("div");
    node.textContent = String(value ?? "");
    return node.innerHTML;
  }

  async function apiRequest(path, options = {}) {
    const init = {
      credentials: "same-origin",
      headers: { Accept: "application/json", ...(options.headers || {}) },
      ...options,
    };
    if (options.body && !(options.body instanceof FormData) && !init.headers["Content-Type"]) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(options.body);
    }
    const response = await fetch(path, init);
    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();
    if (!response.ok) {
      const detail =
        typeof payload === "string"
          ? payload
          : payload.detail || payload.error || JSON.stringify(payload);
      throw new Error(detail);
    }
    return payload;
  }

  function parseJsonText(value, defaultValue) {
    const raw = String(value || "").trim();
    if (!raw) {
      return defaultValue;
    }
    return JSON.parse(raw);
  }

  function parsePageData() {
    const node = document.getElementById("ursa-page-data");
    if (!node) {
      return {};
    }
    try {
      return JSON.parse(node.textContent || "{}");
    } catch (_error) {
      return {};
    }
  }

  function formToObject(form) {
    const data = new FormData(form);
    const result = {};
    for (const [key, value] of data.entries()) {
      result[key] = value;
    }
    return result;
  }

  function bindJsonForm(selector, onSubmit) {
    const form = document.querySelector(selector);
    if (!form) {
      return;
    }
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      try {
        await onSubmit(form);
      } catch (error) {
        showToast("error", error.message);
      }
    });
  }

  function bindClick(selector, handler) {
    document.querySelectorAll(selector).forEach((element) => {
      element.addEventListener("click", async (event) => {
        event.preventDefault();
        try {
          await handler(element, event);
        } catch (error) {
          showToast("error", error.message);
        }
      });
    });
  }

  function copyText(text, successMessage = "Copied to clipboard") {
    navigator.clipboard.writeText(String(text || "")).then(() => {
      showToast("success", successMessage);
    });
  }

  function sortableCellValue(cell) {
    const raw = String(cell?.textContent || "").trim();
    if (!raw) {
      return { type: "blank", value: "" };
    }
    const numeric = raw.replace(/[$,%]/g, "").replace(/,/g, "");
    if (/^-?\d+(\.\d+)?$/.test(numeric)) {
      return { type: "number", value: Number(numeric) };
    }
    const timestamp = Date.parse(raw);
    if (!Number.isNaN(timestamp) && /\d{4}-\d{2}-\d{2}|\d{1,2}\/\d{1,2}\/\d{2,4}/.test(raw)) {
      return { type: "date", value: timestamp };
    }
    return { type: "text", value: raw.toLocaleLowerCase() };
  }

  function compareSortableValues(left, right) {
    if (left.type === "blank" && right.type !== "blank") return 1;
    if (right.type === "blank" && left.type !== "blank") return -1;
    if (left.type === right.type) {
      if (left.value < right.value) return -1;
      if (left.value > right.value) return 1;
      return 0;
    }
    return String(left.value).localeCompare(String(right.value));
  }

  function sortableHeaderLabel(header) {
    return String(header?.textContent || "").trim();
  }

  function canSortHeader(header) {
    if (!header || header.dataset.sortable === "false") return false;
    if (!sortableHeaderLabel(header)) return false;
    return !header.querySelector("button, input, select, textarea, a");
  }

  function sortTableByColumn(table, header, columnIndex) {
    const tbody = table.tBodies[0];
    if (!tbody) return;
    const current = header.getAttribute("aria-sort") === "ascending" ? "ascending" : "";
    const nextDirection = current === "ascending" ? "descending" : "ascending";
    const multiplier = nextDirection === "ascending" ? 1 : -1;
    const rows = Array.from(tbody.rows);
    rows.sort((leftRow, rightRow) => {
      const left = sortableCellValue(leftRow.cells[columnIndex]);
      const right = sortableCellValue(rightRow.cells[columnIndex]);
      return compareSortableValues(left, right) * multiplier;
    });
    rows.forEach((row) => tbody.appendChild(row));
    Array.from(table.tHead?.querySelectorAll("th") || []).forEach((candidate) => {
      candidate.setAttribute("aria-sort", "none");
      const indicator = candidate.querySelector(".sortable-indicator");
      if (indicator) indicator.textContent = "";
    });
    header.setAttribute("aria-sort", nextDirection);
    const indicator = header.querySelector(".sortable-indicator");
    if (indicator) indicator.textContent = nextDirection === "ascending" ? "▲" : "▼";
  }

  function enhanceSortableTable(table) {
    if (!table || table.dataset.sortableInitialized === "true") return;
    const headerRow = table.tHead?.rows?.[0];
    const tbody = table.tBodies?.[0];
    if (!headerRow || !tbody || tbody.rows.length < 1) return;
    table.dataset.sortableInitialized = "true";
    table.classList.add("sortable-table");
    Array.from(headerRow.cells).forEach((header, columnIndex) => {
      if (!canSortHeader(header)) return;
      const label = sortableHeaderLabel(header);
      header.setAttribute("aria-sort", "none");
      header.innerHTML = "";
      const button = document.createElement("button");
      button.type = "button";
      button.className = "sortable-header";
      button.innerHTML = `<span>${escapeHtml(label)}</span><span class="sortable-indicator" aria-hidden="true"></span>`;
      button.addEventListener("click", () => sortTableByColumn(table, header, columnIndex));
      header.appendChild(button);
    });
  }

  let sortableObserver = null;

  function initSortableTables(root = document) {
    const scope = root instanceof Element ? root : document;
    if (scope.matches?.("table")) {
      enhanceSortableTable(scope);
    }
    scope.querySelectorAll("table").forEach((table) => enhanceSortableTable(table));
    if (!sortableObserver && document.body) {
      sortableObserver = new MutationObserver((mutations) => {
        mutations.forEach((mutation) => {
          mutation.addedNodes.forEach((node) => {
            if (!(node instanceof Element)) return;
            if (node.matches("table")) {
              enhanceSortableTable(node);
            }
            node.querySelectorAll?.("table").forEach((table) => enhanceSortableTable(table));
          });
        });
      });
      sortableObserver.observe(document.body, { childList: true, subtree: true });
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    initMenu();
    initUserMenu();
    document.querySelectorAll(".modal .modal-close").forEach((button) => {
      button.addEventListener("click", () => {
        const modal = button.closest(".modal, .modal-overlay");
        if (modal?.id) {
          closeModal(modal.id);
        }
      });
    });
    document.querySelectorAll(".modal-overlay, .modal").forEach((element) => {
      element.addEventListener("click", (event) => {
        if (event.target === element && element.id) {
          closeModal(element.id);
        }
      });
    });
    initSortableTables();
  });

  window.showToast = showToast;
  window.showLoading = showLoading;
  window.hideLoading = hideLoading;
  window.showModal = showModal;
  window.closeModal = closeModal;
  window.escapeHtml = escapeHtml;

  return {
    apiRequest,
    bindClick,
    bindJsonForm,
    closeModal,
    copyText,
    escapeHtml,
    formToObject,
    parseJsonText,
    parsePageData,
    showLoading,
    showModal,
    showToast,
    hideLoading,
    initSortableTables,
  };
})();
