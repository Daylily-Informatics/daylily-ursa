const pageConfig = JSON.parse(document.getElementById("ursa-page-config").textContent);
const controlsEl = document.getElementById("page-controls");
const contentEl = document.getElementById("page-content");
const flashEl = document.getElementById("flash");

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function setFlash(message, type = "success") {
  if (!message) {
    flashEl.className = "flash";
    flashEl.textContent = "";
    return;
  }
  flashEl.className = `flash visible ${type}`;
  flashEl.textContent = message;
}

async function apiRequest(path, options = {}) {
  const init = {
    credentials: "same-origin",
    headers: { Accept: "application/json", ...(options.headers || {}) },
    ...options,
  };
  if (options.body && !init.headers["Content-Type"]) {
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

function table(columns, rows) {
  return `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join("")}</tr>
        </thead>
        <tbody>
          ${rows.length ? rows.join("") : `<tr><td colspan="${columns.length}">No records</td></tr>`}
        </tbody>
      </table>
    </div>
  `;
}

function cardGrid(cards) {
  return `<div class="card-grid">${cards.join("")}</div>`;
}

function miniCard(label, value, body = "") {
  return `
    <article class="mini-card">
      <p class="eyebrow">${escapeHtml(label)}</p>
      <h3>${escapeHtml(value)}</h3>
      ${body ? `<p class="muted">${escapeHtml(body)}</p>` : ""}
    </article>
  `;
}

function jsonBlock(value) {
  return `<pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>`;
}

function parseJsonInput(value, fallback = {}) {
  const raw = String(value || "").trim();
  if (!raw) {
    return fallback;
  }
  return JSON.parse(raw);
}

function bindForm(id, handler) {
  const form = document.getElementById(id);
  if (!form) {
    return;
  }
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    setFlash("");
    try {
      await handler(new FormData(form));
    } catch (error) {
      setFlash(error.message, "error");
    }
  });
}

function bindClicks(selector, handler) {
  document.querySelectorAll(selector).forEach((element) => {
    element.addEventListener("click", async (event) => {
      event.preventDefault();
      setFlash("");
      try {
        await handler(element);
      } catch (error) {
        setFlash(error.message, "error");
      }
    });
  });
}

async function loadUserToken(tokenEuid) {
  const tokens = await apiRequest("/api/v1/user-tokens");
  const token = tokens.find((item) => item.token_euid === tokenEuid);
  if (!token) {
    throw new Error(`User token not found: ${tokenEuid}`);
  }
  return token;
}

async function renderDashboard() {
  const [me, worksets, analyses, tokens] = await Promise.all([
    apiRequest("/api/v1/me"),
    apiRequest("/api/v1/worksets"),
    apiRequest("/api/v1/analyses"),
    apiRequest("/api/v1/user-tokens"),
  ]);

  let adminCards = "";
  if (pageConfig.is_admin) {
    const [clusters, jobs] = await Promise.all([
      apiRequest("/api/v1/clusters"),
      apiRequest("/api/v1/clusters/jobs"),
    ]);
    adminCards = miniCard("Clusters", String((clusters.items || []).length), "Admin-only cluster inventory")
      + miniCard("Cluster Jobs", String(jobs.length), "TapDB-backed job history");
  }

  controlsEl.innerHTML = cardGrid([
    miniCard("User", me.display_name || me.email || me.user_id, me.organization || me.atlas_tenant_id),
    miniCard("Worksets", String(worksets.length), "GUI-facing orchestration units"),
    miniCard("Analyses", String(analyses.length), "Versioned analysis API surface"),
    miniCard("User Tokens", String(tokens.length), "Self-service bearer credentials"),
    adminCards,
  ]);
  contentEl.innerHTML = `
    <h2>Current Context</h2>
    ${jsonBlock(me)}
  `;
}

async function renderWorksets() {
  const worksets = await apiRequest("/api/v1/worksets");
  controlsEl.innerHTML = `
    <h2>Create Workset</h2>
    <form id="workset-create-form" class="stack-form">
      <input name="name" placeholder="Tumor batch" required>
      <textarea name="artifact_set_euids" placeholder='["AS-1", "AS-2"]'></textarea>
      <textarea name="metadata" placeholder='{"project":"pilot"}'></textarea>
      <button class="button primary" type="submit">Create Workset</button>
    </form>
  `;
  contentEl.innerHTML = `
    <h2>Worksets</h2>
    ${table(
      ["Name", "EUID", "State", "Artifacts", "Manifests", "Analyses"],
      worksets.map(
        (item) => `
          <tr>
            <td><a href="/worksets/${encodeURIComponent(item.workset_euid)}">${escapeHtml(item.name)}</a></td>
            <td>${escapeHtml(item.workset_euid)}</td>
            <td>${escapeHtml(item.state)}</td>
            <td>${escapeHtml(item.artifact_set_euids.join(", "))}</td>
            <td>${item.manifests.length}</td>
            <td>${item.analysis_euids.length}</td>
          </tr>
        `
      )
    )}
  `;
  bindForm("workset-create-form", async (formData) => {
    await apiRequest("/api/v1/worksets", {
      method: "POST",
      body: {
        name: formData.get("name"),
        artifact_set_euids: parseJsonInput(formData.get("artifact_set_euids"), []),
        metadata: parseJsonInput(formData.get("metadata"), {}),
      },
    });
    setFlash("Workset created");
    await renderWorksets();
  });
}

async function renderWorksetDetail() {
  const detail = await apiRequest(`/api/v1/worksets/${encodeURIComponent(pageConfig.detail_id)}`);
  controlsEl.innerHTML = "";
  contentEl.innerHTML = `
    <h2>${escapeHtml(detail.name)}</h2>
    ${cardGrid([
      miniCard("State", detail.state, detail.workset_euid),
      miniCard("Artifacts", String(detail.artifact_set_euids.length)),
      miniCard("Manifests", String(detail.manifests.length)),
      miniCard("Analyses", String(detail.analysis_euids.length)),
    ])}
    <h2>Payload</h2>
    ${jsonBlock(detail)}
  `;
}

async function renderManifests() {
  const manifests = await apiRequest("/api/v1/manifests");
  controlsEl.innerHTML = `
    <h2>Create Manifest</h2>
    <form id="manifest-create-form" class="stack-form">
      <input name="workset_euid" placeholder="WS-..." required>
      <input name="name" placeholder="manifest-01" required>
      <input name="artifact_set_euid" placeholder="AS-..." required>
      <textarea name="artifact_euids" placeholder='["AT-1", "AT-2"]'></textarea>
      <textarea name="metadata" placeholder='{"source":"gui"}'></textarea>
      <button class="button primary" type="submit">Create Manifest</button>
    </form>
  `;
  contentEl.innerHTML = `
    <h2>Manifests</h2>
    ${table(
      ["Name", "Manifest EUID", "Workset", "Artifact Set", "Artifacts"],
      manifests.map(
        (item) => `
          <tr>
            <td><a href="/manifests/${encodeURIComponent(item.manifest_euid)}">${escapeHtml(item.name)}</a></td>
            <td>${escapeHtml(item.manifest_euid)}</td>
            <td>${escapeHtml(item.workset_euid)}</td>
            <td>${escapeHtml(item.artifact_set_euid)}</td>
            <td>${escapeHtml(item.artifact_euids.join(", "))}</td>
          </tr>
        `
      )
    )}
  `;
  bindForm("manifest-create-form", async (formData) => {
    await apiRequest("/api/v1/manifests", {
      method: "POST",
      body: {
        workset_euid: formData.get("workset_euid"),
        name: formData.get("name"),
        artifact_set_euid: formData.get("artifact_set_euid"),
        artifact_euids: parseJsonInput(formData.get("artifact_euids"), []),
        metadata: parseJsonInput(formData.get("metadata"), {}),
      },
    });
    setFlash("Manifest created");
    await renderManifests();
  });
}

async function renderManifestDetail() {
  const detail = await apiRequest(`/api/v1/manifests/${encodeURIComponent(pageConfig.detail_id)}`);
  controlsEl.innerHTML = "";
  contentEl.innerHTML = `
    <h2>${escapeHtml(detail.name)}</h2>
    ${jsonBlock(detail)}
  `;
}

async function renderAnalyses() {
  const analyses = await apiRequest("/api/v1/analyses");
  controlsEl.innerHTML = `
    <h2>Analyses</h2>
    <p class="muted">Run ingest still happens through the write-key API. The GUI uses versioned read/review/return flows here.</p>
  `;
  contentEl.innerHTML = table(
    ["Analysis", "Type", "State", "Review", "Result", "Workset"],
    analyses.map(
      (item) => `
        <tr>
          <td><a href="/analyses/${encodeURIComponent(item.analysis_euid)}">${escapeHtml(item.analysis_euid)}</a></td>
          <td>${escapeHtml(item.analysis_type)}</td>
          <td>${escapeHtml(item.state)}</td>
          <td>${escapeHtml(item.review_state)}</td>
          <td>${escapeHtml(item.result_status)}</td>
          <td>${escapeHtml(item.workset_euid || "")}</td>
        </tr>
      `
    )
  );
}

async function renderAnalysisDetail() {
  const detail = await apiRequest(`/api/v1/analyses/${encodeURIComponent(pageConfig.detail_id)}`);
  controlsEl.innerHTML = `
    <h2>Review</h2>
    <form id="analysis-review-form" class="split-form">
      <select name="review_state">
        <option value="APPROVED">APPROVED</option>
        <option value="REJECTED">REJECTED</option>
        <option value="PENDING">PENDING</option>
      </select>
      <input name="reviewer" placeholder="reviewer id">
      <textarea name="notes" placeholder="review notes"></textarea>
      <button class="button primary" type="submit">Submit Review</button>
    </form>
    <h2>Return</h2>
    <form id="analysis-return-form" class="stack-form">
      <textarea name="result_payload" placeholder='{"status":"ok"}'></textarea>
      <input name="idempotency_key" placeholder="return-key-001" required>
      <button class="button primary" type="submit">Return Result</button>
    </form>
  `;
  contentEl.innerHTML = `
    <h2>${escapeHtml(detail.analysis_euid)}</h2>
    ${jsonBlock(detail)}
  `;
  bindForm("analysis-review-form", async (formData) => {
    await apiRequest(`/api/v1/analyses/${encodeURIComponent(pageConfig.detail_id)}/review`, {
      method: "POST",
      body: {
        review_state: formData.get("review_state"),
        reviewer: formData.get("reviewer"),
        notes: formData.get("notes"),
      },
    });
    setFlash("Analysis review updated");
    await renderAnalysisDetail();
  });
  bindForm("analysis-return-form", async (formData) => {
    await apiRequest(`/api/v1/analyses/${encodeURIComponent(pageConfig.detail_id)}/return`, {
      method: "POST",
      headers: { "Idempotency-Key": formData.get("idempotency_key") },
      body: {
        result_payload: parseJsonInput(formData.get("result_payload"), {}),
        result_status: "COMPLETED",
      },
    });
    setFlash("Analysis return submitted");
    await renderAnalysisDetail();
  });
}

async function renderArtifacts() {
  controlsEl.innerHTML = `
    <div class="detail-grid">
      <div>
        <h2>Import to Dewey</h2>
        <form id="artifact-import-form" class="stack-form">
          <input name="artifact_type" placeholder="fastq" required>
          <input name="storage_uri" placeholder="s3://bucket/key" required>
          <textarea name="metadata" placeholder='{"source":"gui"}'></textarea>
          <button class="button primary" type="submit">Import Artifact</button>
        </form>
      </div>
      <div>
        <h2>Resolve Artifact</h2>
        <form id="artifact-resolve-form" class="stack-form">
          <input name="artifact_euid" placeholder="AT-...">
          <input name="artifact_set_euid" placeholder="AS-...">
          <button class="button primary" type="submit">Resolve</button>
        </form>
      </div>
    </div>
  `;
  contentEl.innerHTML = `<h2>Artifact Results</h2><div id="artifact-result"></div>`;
  bindForm("artifact-import-form", async (formData) => {
    const result = await apiRequest("/api/v1/artifacts/import", {
      method: "POST",
      body: {
        artifact_type: formData.get("artifact_type"),
        storage_uri: formData.get("storage_uri"),
        metadata: parseJsonInput(formData.get("metadata"), {}),
      },
    });
    document.getElementById("artifact-result").innerHTML = jsonBlock(result);
    setFlash("Artifact imported into Dewey");
  });
  bindForm("artifact-resolve-form", async (formData) => {
    const artifactEuid = String(formData.get("artifact_euid") || "").trim();
    const artifactSetEuid = String(formData.get("artifact_set_euid") || "").trim();
    const result = await apiRequest("/api/v1/artifacts/resolve", {
      method: "POST",
      body: artifactEuid
        ? { artifact_euid: artifactEuid }
        : { artifact_set_euid: artifactSetEuid },
    });
    document.getElementById("artifact-result").innerHTML = jsonBlock(result);
    setFlash("Artifact resolution completed");
  });
}

async function renderTokens(createdToken = null) {
  const tokens = await apiRequest("/api/v1/user-tokens");
  controlsEl.innerHTML = `
    <h2>Create Token</h2>
    <form id="user-token-form" class="split-form">
      <input name="token_name" placeholder="gui token" required>
      <input name="scope" value="internal_rw" required>
      <input name="expires_in_days" type="number" value="30" min="1" max="3650">
      <input name="note" placeholder="optional note">
      <button class="button primary" type="submit">Create Token</button>
    </form>
    <div id="token-create-result">${createdToken ? jsonBlock(createdToken) : ""}</div>
  `;
  contentEl.innerHTML = `
    <h2>Tokens</h2>
    ${table(
      ["Name", "EUID", "Scope", "Status", "Expires", "Last Used", "Usage", "Action"],
      tokens.map(
        (item) => `
          <tr>
            <td><a href="/tokens/${encodeURIComponent(item.token_euid)}">${escapeHtml(item.token_name)}</a></td>
            <td>${escapeHtml(item.token_euid)}</td>
            <td>${escapeHtml(item.scope)}</td>
            <td>${escapeHtml(item.status)}</td>
            <td>${escapeHtml(item.expires_at)}</td>
            <td>${escapeHtml(item.last_used_at || "")}</td>
            <td><a class="button ghost" href="/tokens/${encodeURIComponent(item.token_euid)}">Usage</a></td>
            <td><button class="button ghost" data-token-revoke="${escapeHtml(item.token_euid)}">Revoke</button></td>
          </tr>
        `
      )
    )}
  `;
  bindForm("user-token-form", async (formData) => {
    const result = await apiRequest("/api/v1/user-tokens", {
      method: "POST",
      body: {
        token_name: formData.get("token_name"),
        scope: formData.get("scope"),
        expires_in_days: Number(formData.get("expires_in_days") || 30),
        note: formData.get("note"),
      },
    });
    setFlash("Token created");
    await renderTokens(result);
  });
  bindClicks("[data-token-revoke]", async (element) => {
    await apiRequest(`/api/v1/user-tokens/${encodeURIComponent(element.dataset.tokenRevoke)}/revoke`, {
      method: "POST",
      body: { note: "revoked from gui" },
    });
    setFlash("Token revoked");
    await renderTokens();
  });
}

async function renderTokenDetail() {
  const [token, usage] = await Promise.all([
    loadUserToken(pageConfig.detail_id),
    apiRequest(`/api/v1/user-tokens/${encodeURIComponent(pageConfig.detail_id)}/usage`),
  ]);
  controlsEl.innerHTML = `
    <div class="detail-grid">
      <div>
        <h2>Token</h2>
        ${cardGrid([
          miniCard("Scope", token.scope, token.token_euid),
          miniCard("Status", token.status, token.token_name),
          miniCard("Last Used", token.last_used_at || "Never"),
          miniCard("Expires", token.expires_at),
        ])}
      </div>
      <div>
        <h2>Actions</h2>
        <p><a class="button ghost" href="/tokens">Back to Tokens</a></p>
        <p><button class="button primary" id="token-detail-revoke">Revoke Token</button></p>
      </div>
    </div>
  `;
  contentEl.innerHTML = `
    <h2>Token Details</h2>
    ${jsonBlock(token)}
    <h2>Usage</h2>
    ${table(
      ["When", "Method", "Endpoint", "Status", "IP", "Request"],
      usage.map(
        (item) => `
          <tr>
            <td>${escapeHtml(item.created_at)}</td>
            <td>${escapeHtml(item.http_method)}</td>
            <td>${escapeHtml(item.endpoint)}</td>
            <td>${escapeHtml(String(item.response_status))}</td>
            <td>${escapeHtml(item.ip_address || "")}</td>
            <td>${escapeHtml(JSON.stringify(item.request_metadata || {}))}</td>
          </tr>
        `
      )
    )}
  `;
  bindClicks("#token-detail-revoke", async () => {
    await apiRequest(`/api/v1/user-tokens/${encodeURIComponent(pageConfig.detail_id)}/revoke`, {
      method: "POST",
      body: { note: "revoked from gui detail" },
    });
    setFlash("Token revoked");
    window.location.assign("/tokens");
  });
}

async function renderClusters() {
  const [clusters, jobs] = await Promise.all([
    apiRequest("/api/v1/clusters"),
    apiRequest("/api/v1/clusters/jobs"),
  ]);
  controlsEl.innerHTML = `
    <h2>Create Ephemeral Cluster</h2>
    <form id="cluster-create-form" class="split-form">
      <input name="cluster_name" placeholder="ursa-pilot-01" required>
      <input name="region_az" placeholder="us-west-2d" required>
      <input name="ssh_key_name" placeholder="omics-key" required>
      <input name="s3_bucket_name" placeholder="ursa-bucket" required>
      <input name="owner_user_id" placeholder="owner user id">
      <input name="contact_email" placeholder="ops@example.com">
      <button class="button primary" type="submit">Queue Cluster Job</button>
    </form>
  `;
  contentEl.innerHTML = `
    <h2>Clusters</h2>
    ${table(
      ["Name", "Region", "Status", "Action"],
      (clusters.items || []).map(
        (item) => `
          <tr>
            <td><a href="/clusters/${encodeURIComponent(item.cluster_name)}?region=${encodeURIComponent(item.region)}">${escapeHtml(item.cluster_name)}</a></td>
            <td>${escapeHtml(item.region)}</td>
            <td>${escapeHtml(item.cluster_status || "")}</td>
            <td><button class="button ghost" data-cluster-delete="${escapeHtml(item.cluster_name)}" data-region="${escapeHtml(item.region)}">Delete</button></td>
          </tr>
        `
      )
    )}
    <h2>Cluster Jobs</h2>
    ${table(
      ["Job", "Cluster", "State", "Updated", "Summary"],
      jobs.map(
        (item) => `
          <tr>
            <td><a href="/clusters/jobs/${encodeURIComponent(item.job_euid)}">${escapeHtml(item.job_euid)}</a></td>
            <td>${escapeHtml(item.cluster_name)}</td>
            <td>${escapeHtml(item.state)}</td>
            <td>${escapeHtml(item.updated_at)}</td>
            <td>${escapeHtml(item.output_summary || "")}</td>
          </tr>
        `
      )
    )}
  `;
  bindForm("cluster-create-form", async (formData) => {
    await apiRequest("/api/v1/clusters", {
      method: "POST",
      body: {
        cluster_name: formData.get("cluster_name"),
        region_az: formData.get("region_az"),
        ssh_key_name: formData.get("ssh_key_name"),
        s3_bucket_name: formData.get("s3_bucket_name"),
        owner_user_id: formData.get("owner_user_id") || null,
        contact_email: formData.get("contact_email") || null,
      },
    });
    setFlash("Cluster job queued");
    await renderClusters();
  });
  bindClicks("[data-cluster-delete]", async (element) => {
    await apiRequest(
      `/api/v1/clusters/${encodeURIComponent(element.dataset.clusterDelete)}?region=${encodeURIComponent(element.dataset.region)}`,
      { method: "DELETE" }
    );
    setFlash("Cluster delete submitted");
    await renderClusters();
  });
}

async function renderClusterDetail() {
  const regionPart = pageConfig.detail_region ? `?region=${encodeURIComponent(pageConfig.detail_region)}` : "";
  const [detail, jobs] = await Promise.all([
    apiRequest(`/api/v1/clusters/${encodeURIComponent(pageConfig.detail_id)}${regionPart}`),
    apiRequest("/api/v1/clusters/jobs"),
  ]);
  const relatedJobs = jobs.filter((item) => item.cluster_name === pageConfig.detail_id);
  controlsEl.innerHTML = `
    <p><button class="button ghost" id="cluster-refresh">Refresh Cluster</button></p>
  `;
  contentEl.innerHTML = `
    <h2>Cluster Snapshot</h2>
    ${jsonBlock(detail)}
    <h2>Related Jobs</h2>
    ${table(
      ["Job", "State", "Updated", "Summary"],
      relatedJobs.map(
        (item) => `
          <tr>
            <td><a href="/clusters/jobs/${encodeURIComponent(item.job_euid)}">${escapeHtml(item.job_euid)}</a></td>
            <td>${escapeHtml(item.state)}</td>
            <td>${escapeHtml(item.updated_at)}</td>
            <td>${escapeHtml(item.output_summary || "")}</td>
          </tr>
        `
      )
    )}
  `;
  bindClicks("#cluster-refresh", async () => {
    await renderClusterDetail();
    setFlash("Cluster detail refreshed");
  });
}

async function renderClusterJobDetail() {
  const detail = await apiRequest(`/api/v1/clusters/jobs/${encodeURIComponent(pageConfig.detail_id)}`);
  controlsEl.innerHTML = `
    <div class="detail-grid">
      <div>
        <h2>Job Status</h2>
        ${cardGrid([
          miniCard("State", detail.state, detail.job_euid),
          miniCard("Cluster", detail.cluster_name, detail.region_az),
          miniCard("Return Code", detail.return_code == null ? "pending" : String(detail.return_code)),
          miniCard("Updated", detail.updated_at),
        ])}
      </div>
      <div>
        <h2>Actions</h2>
        <p><button class="button ghost" id="cluster-job-refresh">Refresh Job</button></p>
        <p><a class="button ghost" href="/clusters/${encodeURIComponent(detail.cluster_name)}?region=${encodeURIComponent(detail.region)}">Open Cluster</a></p>
      </div>
    </div>
  `;
  contentEl.innerHTML = `
    <h2>Job Payload</h2>
    ${jsonBlock(detail)}
    <h2>Events</h2>
    ${table(
      ["When", "Type", "Status", "Summary"],
      (detail.events || []).map(
        (item) => `
          <tr>
            <td>${escapeHtml(item.created_at)}</td>
            <td>${escapeHtml(item.event_type)}</td>
            <td>${escapeHtml(item.status)}</td>
            <td>${escapeHtml(item.summary)}</td>
          </tr>
        `
      )
    )}
  `;
  bindClicks("#cluster-job-refresh", async () => {
    await renderClusterJobDetail();
    setFlash("Cluster job refreshed");
  });
}

async function renderAdminTokens(createdToken = null) {
  const tokens = await apiRequest("/api/v1/admin/user-tokens?owner_user_id=*");
  controlsEl.innerHTML = `
    <div class="detail-grid">
      <div>
        <h2>Search Atlas Users</h2>
        <form id="admin-user-search-form" class="split-form">
          <input name="search" placeholder="alice@example.com">
          <button class="button primary" type="submit">Search</button>
        </form>
        <div id="admin-user-results"></div>
      </div>
      <div>
        <h2>Create Admin Token</h2>
        <form id="admin-token-form" class="stack-form">
          <input id="admin-owner-user-id" name="owner_user_id" placeholder="owner user id" required>
          <input name="token_name" placeholder="client bootstrap token" required>
          <input name="scope" value="internal_rw" required>
          <input name="expires_in_days" type="number" value="30">
          <input name="note" placeholder="note">
          <button class="button primary" type="submit">Create Token</button>
        </form>
        <div id="admin-token-result">${createdToken ? jsonBlock(createdToken) : ""}</div>
      </div>
    </div>
  `;
  contentEl.innerHTML = `
    <h2>All User Tokens</h2>
    ${table(
      ["Owner", "Token", "Scope", "Status", "Client Registration", "Last Used", "Action"],
      tokens.map(
        (item) => `
          <tr>
            <td>${escapeHtml(item.owner_user_id)}</td>
            <td>${escapeHtml(item.token_euid)}</td>
            <td>${escapeHtml(item.scope)}</td>
            <td>${escapeHtml(item.status)}</td>
            <td>${escapeHtml(item.client_registration_euid || "")}</td>
            <td>${escapeHtml(item.last_used_at || "")}</td>
            <td><button class="button ghost" data-admin-token-revoke="${escapeHtml(item.token_euid)}">Revoke</button></td>
          </tr>
        `
      )
    )}
  `;
  bindForm("admin-user-search-form", async (formData) => {
    const users = await apiRequest(`/api/v1/admin/users?search=${encodeURIComponent(formData.get("search") || "")}`);
    document.getElementById("admin-user-results").innerHTML = table(
      ["User", "Tenant", "Roles", "Select"],
      users.map(
        (item) => `
          <tr>
            <td>${escapeHtml(item.email || item.user_id)}</td>
            <td>${escapeHtml(item.atlas_tenant_id)}</td>
            <td>${escapeHtml((item.roles || []).join(", "))}</td>
            <td><button class="button ghost" data-select-owner="${escapeHtml(item.user_id)}">Use</button></td>
          </tr>
        `
      )
    );
    bindClicks("[data-select-owner]", async (element) => {
      document.getElementById("admin-owner-user-id").value = element.dataset.selectOwner;
      setFlash(`Selected ${element.dataset.selectOwner}`);
    });
  });
  bindForm("admin-token-form", async (formData) => {
    const result = await apiRequest("/api/v1/admin/user-tokens", {
      method: "POST",
      body: {
        owner_user_id: formData.get("owner_user_id"),
        token_name: formData.get("token_name"),
        scope: formData.get("scope"),
        expires_in_days: Number(formData.get("expires_in_days") || 30),
        note: formData.get("note"),
      },
    });
    setFlash("Admin token created");
    await renderAdminTokens(result);
  });
  bindClicks("[data-admin-token-revoke]", async (element) => {
    await apiRequest(`/api/v1/admin/user-tokens/${encodeURIComponent(element.dataset.adminTokenRevoke)}/revoke`, {
      method: "POST",
      body: { note: "revoked from admin gui" },
    });
    setFlash("Admin token revoked");
    await renderAdminTokens();
  });
}

async function renderAdminClients() {
  const clients = await apiRequest("/api/v1/admin/client-registrations");
  controlsEl.innerHTML = `
    <h2>Create Client Registration</h2>
    <form id="client-registration-form" class="stack-form">
      <input name="client_name" placeholder="dewey-ingest-client" required>
      <input name="owner_user_id" placeholder="owner user id" required>
      <textarea name="scopes" placeholder='["internal_rw"]'></textarea>
      <textarea name="metadata" placeholder='{"purpose":"integration"}'></textarea>
      <button class="button primary" type="submit">Create Client Registration</button>
    </form>
  `;
  contentEl.innerHTML = `
    <h2>Client Registrations</h2>
    ${table(
      ["Client", "Owner", "Scopes", "State"],
      clients.map(
        (item) => `
          <tr>
            <td><a href="/admin/clients/${encodeURIComponent(item.client_registration_euid)}">${escapeHtml(item.client_name)}</a></td>
            <td>${escapeHtml(item.owner_user_id)}</td>
            <td>${escapeHtml(item.scopes.join(", "))}</td>
            <td>${escapeHtml(item.state)}</td>
          </tr>
        `
      )
    )}
  `;
  bindForm("client-registration-form", async (formData) => {
    await apiRequest("/api/v1/admin/client-registrations", {
      method: "POST",
      body: {
        client_name: formData.get("client_name"),
        owner_user_id: formData.get("owner_user_id"),
        scopes: parseJsonInput(formData.get("scopes"), []),
        metadata: parseJsonInput(formData.get("metadata"), {}),
      },
    });
    setFlash("Client registration created");
    await renderAdminClients();
  });
}

async function renderAdminClientDetail(createdToken = null) {
  const [registration, tokens] = await Promise.all([
    apiRequest(`/api/v1/admin/client-registrations/${encodeURIComponent(pageConfig.detail_id)}`),
    apiRequest(`/api/v1/admin/client-registrations/${encodeURIComponent(pageConfig.detail_id)}/tokens`),
  ]);
  controlsEl.innerHTML = `
    <h2>Issue Client Token</h2>
    <form id="client-token-form" class="split-form">
      <input name="token_name" placeholder="client bootstrap token" required>
      <input name="scope" value="${escapeHtml((registration.scopes || [])[0] || "internal_rw")}" required>
      <input name="expires_in_days" type="number" value="30">
      <input name="note" placeholder="optional note">
      <button class="button primary" type="submit">Create Client Token</button>
    </form>
    <div id="client-token-result">${createdToken ? jsonBlock(createdToken) : ""}</div>
  `;
  contentEl.innerHTML = `
    <h2>${escapeHtml(registration.client_name)}</h2>
    ${jsonBlock(registration)}
    <h2>Client Tokens</h2>
    ${table(
      ["Token", "Scope", "Status", "Expires", "Last Used", "Action"],
      tokens.map(
        (item) => `
          <tr>
            <td>${escapeHtml(item.token_euid)}</td>
            <td>${escapeHtml(item.scope)}</td>
            <td>${escapeHtml(item.status)}</td>
            <td>${escapeHtml(item.expires_at)}</td>
            <td>${escapeHtml(item.last_used_at || "")}</td>
            <td><button class="button ghost" data-client-token-revoke="${escapeHtml(item.token_euid)}">Revoke</button></td>
          </tr>
        `
      )
    )}
  `;
  bindForm("client-token-form", async (formData) => {
    const result = await apiRequest(
      `/api/v1/admin/client-registrations/${encodeURIComponent(pageConfig.detail_id)}/tokens`,
      {
        method: "POST",
        body: {
          token_name: formData.get("token_name"),
          scope: formData.get("scope"),
          expires_in_days: Number(formData.get("expires_in_days") || 30),
          note: formData.get("note"),
        },
      }
    );
    setFlash("Client token created");
    await renderAdminClientDetail(result);
  });
  bindClicks("[data-client-token-revoke]", async (element) => {
    await apiRequest(`/api/v1/admin/user-tokens/${encodeURIComponent(element.dataset.clientTokenRevoke)}/revoke`, {
      method: "POST",
      body: { note: `revoked from client registration ${pageConfig.detail_id}` },
    });
    setFlash("Client token revoked");
    await renderAdminClientDetail();
  });
}

async function renderCurrentView() {
  switch (pageConfig.view) {
    case "dashboard":
      return renderDashboard();
    case "worksets":
      return renderWorksets();
    case "workset_detail":
      return renderWorksetDetail();
    case "manifests":
      return renderManifests();
    case "manifest_detail":
      return renderManifestDetail();
    case "analyses":
      return renderAnalyses();
    case "analysis_detail":
      return renderAnalysisDetail();
    case "artifacts":
      return renderArtifacts();
    case "tokens":
      return renderTokens();
    case "token_detail":
      return renderTokenDetail();
    case "clusters":
      return renderClusters();
    case "cluster_detail":
      return renderClusterDetail();
    case "cluster_job_detail":
      return renderClusterJobDetail();
    case "admin_tokens":
      return renderAdminTokens();
    case "admin_clients":
      return renderAdminClients();
    case "admin_client_detail":
      return renderAdminClientDetail();
    default:
      controlsEl.innerHTML = "";
      contentEl.innerHTML = "<p class='muted'>Unknown view.</p>";
  }
}

document.addEventListener("DOMContentLoaded", async () => {
  try {
    await renderCurrentView();
  } catch (error) {
    setFlash(error.message, "error");
    contentEl.innerHTML = `<pre>${escapeHtml(error.stack || error.message)}</pre>`;
  }
});
