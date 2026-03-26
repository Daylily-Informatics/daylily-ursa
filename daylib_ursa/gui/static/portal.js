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

  function parseJsonText(value, fallback) {
    const raw = String(value || "").trim();
    if (!raw) {
      return fallback;
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
  };
})();
