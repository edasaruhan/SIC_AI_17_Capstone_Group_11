"use strict";

const API_KEY_STORAGE = "food-demand-api-key";
const HISTORY_STORAGE = "food-demand-session-history";

const displayLabels = {
  category: {
    Beverages: "İçecek", Biryani: "Biryani", Desert: "Tatlı", Extras: "Ek ürün",
    Fish: "Balık", "Other Snacks": "Diğer atıştırmalık", Pasta: "Makarna",
    Pizza: "Pizza", "Rice Bowl": "Pilav kasesi", Salad: "Salata",
    Sandwich: "Sandviç", Seafood: "Deniz ürünü", Soup: "Çorba", Starters: "Başlangıç",
  },
  cuisine: {
    Continental: "Dünya mutfağı", Indian: "Hint mutfağı",
    Italian: "İtalyan mutfağı", Thai: "Tayland mutfağı",
  },
  center_type: {
    TYPE_A: "Tip A (veri seti kodu)",
    TYPE_B: "Tip B (veri seti kodu)",
    TYPE_C: "Tip C (veri seti kodu)",
  },
};

const state = {
  apiKey: sessionStorage.getItem(API_KEY_STORAGE) || "",
  modelInfo: null,
  history: readSessionHistory(),
  batchRows: [],
  batchResults: [],
};

const views = {
  single: document.querySelector("#single-view"),
  quick: document.querySelector("#quick-view"),
  batch: document.querySelector("#batch-view"),
  monitoring: document.querySelector("#monitoring-view"),
};

const requiredCsvColumns = [
  "week", "center_id", "meal_id", "checkout_price", "base_price",
  "emailer_for_promotion", "homepage_featured", "category", "cuisine",
  "center_type", "city_code", "region_code", "op_area", "recent_orders",
];

function readSessionHistory() {
  try {
    const parsed = JSON.parse(sessionStorage.getItem(HISTORY_STORAGE) || "[]");
    return Array.isArray(parsed) ? parsed.slice(0, 20) : [];
  } catch {
    return [];
  }
}

function saveSessionHistory() {
  sessionStorage.setItem(HISTORY_STORAGE, JSON.stringify(state.history.slice(0, 20)));
}

function formatNumber(value, maximumFractionDigits = 0) {
  return new Intl.NumberFormat("tr-TR", { maximumFractionDigits }).format(Number(value));
}

function formatPercent(value, digits = 2) {
  return new Intl.NumberFormat("tr-TR", {
    style: "percent",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  }).format(Number(value));
}

function formatDateTime(value = new Date()) {
  return new Intl.DateTimeFormat("tr-TR", {
    dateStyle: "short",
    timeStyle: "short",
  }).format(value instanceof Date ? value : new Date(value));
}

function showToast(message, type = "success") {
  const region = document.querySelector("#toast-region");
  const toast = document.createElement("div");
  toast.className = `toast ${type === "error" ? "error" : ""}`;
  toast.textContent = message;
  region.appendChild(toast);
  window.setTimeout(() => toast.remove(), 4200);
}

function showMessage(element, message, type = "error") {
  element.textContent = message;
  element.classList.remove("hidden", "success");
  if (type === "success") element.classList.add("success");
}

function hideMessage(element) {
  element.textContent = "";
  element.classList.add("hidden");
  element.classList.remove("success");
}

async function readError(response) {
  try {
    const body = await response.json();
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg).join("; ");
    }
    return body.detail || `İstek başarısız (${response.status})`;
  } catch {
    return `İstek başarısız (${response.status})`;
  }
}

async function apiFetch(path, options = {}) {
  if (!state.apiKey) {
    openApiDialog("Devam etmek için bağlantı anahtarını girin.");
    throw new Error("Bağlantı anahtarı girilmedi.");
  }
  const headers = new Headers(options.headers || {});
  headers.set("X-API-Key", state.apiKey);
  if (options.body && !headers.has("Content-Type")) headers.set("Content-Type", "application/json");
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) throw new Error(await readError(response));
  return response;
}

async function checkHealth() {
  const badge = document.querySelector("#health-badge");
  const label = document.querySelector("#health-label");
  try {
    const response = await fetch("/health", { cache: "no-store" });
    if (!response.ok) throw new Error("Servis yanıt vermedi");
    const health = await response.json();
    badge.className = "status-badge";
    label.textContent = health.status === "ok" ? "Sistem hazır" : "Sistem durumu belirsiz";
    document.querySelector("#api-version").textContent = `Sürüm ${health.version}`;
  } catch {
    badge.className = "status-badge status-error";
    label.textContent = "Sistem çevrimdışı";
    document.querySelector("#api-version").textContent = "Sürüm —";
  }
}

function setView(name) {
  const selected = views[name] ? name : "single";
  Object.entries(views).forEach(([viewName, element]) => {
    element.classList.toggle("hidden", viewName !== selected);
  });
  document.querySelectorAll(".nav-button").forEach((button) => {
    const active = button.dataset.view === selected;
    button.classList.toggle("active", active);
    button.setAttribute("aria-current", active ? "page" : "false");
  });
  const hashes = { single: "tekli", quick: "saha", batch: "toplu", monitoring: "izleme" };
  history.replaceState(null, "", `#${hashes[selected]}`);
  if (selected === "monitoring") refreshMonitoring();
}

function viewFromHash() {
  if (location.hash === "#saha") return "quick";
  if (location.hash === "#toplu") return "batch";
  if (location.hash === "#izleme") return "monitoring";
  return "single";
}

function updateConnectionBanner() {
  document.querySelector("#connection-banner").classList.toggle("hidden", Boolean(state.apiKey));
  document.querySelector("#open-api-settings").textContent = state.apiKey ? "Sistem bağlı" : "Bağlantı ayarı";
}

function openApiDialog(message = "") {
  const dialog = document.querySelector("#api-dialog");
  document.querySelector("#api-key-input").value = state.apiKey;
  const messageBox = document.querySelector("#api-key-message");
  if (message) showMessage(messageBox, message);
  else hideMessage(messageBox);
  if (!dialog.open) dialog.showModal();
  window.setTimeout(() => document.querySelector("#api-key-input").focus(), 0);
}

async function saveApiKey() {
  const input = document.querySelector("#api-key-input");
  const message = document.querySelector("#api-key-message");
  const button = document.querySelector("#save-api-key");
  const candidate = input.value.trim();
  if (!candidate) {
    showMessage(message, "Bağlantı anahtarı boş bırakılamaz.");
    return;
  }
  button.disabled = true;
  button.textContent = "Doğrulanıyor…";
  const previous = state.apiKey;
  state.apiKey = candidate;
  try {
    await loadModelInfo();
    sessionStorage.setItem(API_KEY_STORAGE, candidate);
    updateConnectionBanner();
    document.querySelector("#api-dialog").close();
    showToast("Sistem bağlantısı doğrulandı.");
    await loadLiveMetrics();
  } catch (error) {
    state.apiKey = previous;
    showMessage(message, error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Kaydet ve bağlantıyı doğrula";
  }
}

async function loadModelInfo() {
  const response = await apiFetch("/model-info", { cache: "no-store" });
  state.modelInfo = await response.json();
  populateSelect("#category-select", state.modelInfo.categories.category, "Beverages", displayLabels.category);
  populateSelect("#cuisine-select", state.modelInfo.categories.cuisine, "Italian", displayLabels.cuisine);
  populateSelect("#center-type-select", state.modelInfo.categories.center_type, "TYPE_B", displayLabels.center_type);
  populateSelect("#quick-category-select", state.modelInfo.categories.category, "Beverages", displayLabels.category);
  populateSelect("#quick-cuisine-select", state.modelInfo.categories.cuisine, "Italian", displayLabels.cuisine);
  populateSelect("#quick-center-type-select", state.modelInfo.categories.center_type, "TYPE_B", displayLabels.center_type);
  renderValidationMetrics();
  return state.modelInfo;
}

function populateSelect(selector, values, preferred, labels = {}) {
  const select = document.querySelector(selector);
  const current = select.value;
  select.replaceChildren();
  values.forEach((value) => {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = labels[value] || value;
    select.appendChild(option);
  });
  const desired = values.includes(current) ? current : values.includes(preferred) ? preferred : values[0];
  select.value = desired;
}

function requestFromForm(form) {
  const data = new FormData(form);
  return {
    week: Number(data.get("week")),
    center_id: Number(data.get("center_id")),
    meal_id: Number(data.get("meal_id")),
    checkout_price: Number(data.get("checkout_price")),
    base_price: Number(data.get("base_price")),
    emailer_for_promotion: data.get("emailer_for_promotion") ? 1 : 0,
    homepage_featured: data.get("homepage_featured") ? 1 : 0,
    category: String(data.get("category")),
    cuisine: String(data.get("cuisine")),
    center_type: String(data.get("center_type")),
    city_code: Number(data.get("city_code")),
    region_code: Number(data.get("region_code")),
    op_area: Number(data.get("op_area")),
    recent_orders: [1, 2, 3, 4].map((index) => Number(data.get(`history_${index}`))),
  };
}

async function submitPrediction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const message = document.querySelector("#single-form-message");
  hideMessage(message);
  if (!form.reportValidity()) return;

  const request = requestFromForm(form);
  const button = document.querySelector("#predict-button");
  button.disabled = true;
  button.textContent = "Hesaplanıyor…";
  try {
    const response = await apiFetch("/predict", {
      method: "POST",
      body: JSON.stringify(request),
    });
    const result = await response.json();
    renderPrediction(request, result);
    recordPrediction(request, result);
    showToast("Üretim önerisi hazırlandı.");
  } catch (error) {
    showMessage(message, error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Üretim önerisini hesapla";
  }
}

function recordPrediction(request, result) {
  state.history.unshift({ createdAt: new Date().toISOString(), request, result });
  state.history = state.history.slice(0, 20);
  saveSessionHistory();
  renderHistory();
  renderQuickHistory();
  renderSessionChart();
}

function fillQuickExample() {
  const form = document.querySelector("#quick-form");
  const values = {
    week: 136, center_id: 10, meal_id: 1062,
    checkout_price: 194.06, base_price: 194.06,
    history_1: 1538, history_2: 700, history_3: 704, history_4: 960,
    category: "Beverages", cuisine: "Italian", center_type: "TYPE_B",
    city_code: 590, region_code: 56, op_area: 6.3,
  };
  Object.entries(values).forEach(([name, value]) => {
    const field = form.elements.namedItem(name);
    if (field) field.value = String(value);
  });
  form.elements.namedItem("emailer_for_promotion").checked = false;
  form.elements.namedItem("homepage_featured").checked = false;
  hideMessage(document.querySelector("#quick-form-message"));
  showToast("Örnek veri yüklendi. Bunlar gerçek işletme verisi değildir.");
}

async function submitQuickPrediction(event) {
  event.preventDefault();
  const form = event.currentTarget;
  const message = document.querySelector("#quick-form-message");
  hideMessage(message);
  if (!form.reportValidity()) return;
  const request = requestFromForm(form);
  const button = document.querySelector("#quick-predict-button");
  button.disabled = true;
  button.textContent = "Hesaplanıyor…";
  try {
    const response = await apiFetch("/predict", { method: "POST", body: JSON.stringify(request) });
    const result = await response.json();
    renderQuickPrediction(request, result);
    recordPrediction(request, result);
    showToast("Üretim önerisi hazırlandı.");
  } catch (error) {
    showMessage(message, error.message);
  } finally {
    button.disabled = false;
    button.textContent = "Gelecek haftayı hesapla";
  }
}

function renderQuickPrediction(request, result) {
  document.querySelector("#quick-result-empty").classList.add("hidden");
  document.querySelector("#quick-result-content").classList.remove("hidden");
  document.querySelector("#quick-predicted-orders").textContent = formatNumber(Math.round(result.predicted_orders));
  document.querySelector("#quick-p-low").textContent = formatNumber(Math.round(result.p_low));
  document.querySelector("#quick-p-high").textContent = formatNumber(Math.round(result.p_high));
  const difference = result.predicted_orders - request.recent_orders.at(-1);
  document.querySelector("#quick-comparison").textContent = `Geçen haftaya göre yaklaşık ${difference >= 0 ? "+" : ""}${formatNumber(Math.round(difference))} porsiyon`;
  const warning = document.querySelector("#quick-warning");
  warning.classList.toggle("hidden", !result.warnings?.length);
  warning.textContent = result.warnings?.length ? "Geçmiş veri yetersiz; öneriyi işletme koşullarıyla birlikte değerlendirin." : "";
  document.querySelector("#quick-version").textContent = `Model ${result.model_version} · ${formatDateTime()}`;
}

function renderQuickHistory() {
  const list = document.querySelector("#quick-recent-list");
  list.replaceChildren();
  if (!state.history.length) {
    const item = document.createElement("li");
    item.textContent = "Henüz tahmin yapılmadı.";
    list.appendChild(item);
    return;
  }
  state.history.slice(0, 3).forEach(({ request, result }) => {
    const item = document.createElement("li");
    const label = document.createElement("span");
    label.textContent = `Yemek ${request.meal_id} · Hafta ${request.week}`;
    const value = document.createElement("strong");
    value.textContent = `${formatNumber(Math.round(result.predicted_orders))} pors.`;
    item.append(label, value);
    list.appendChild(item);
  });
}

function renderPrediction(request, result) {
  document.querySelector("#result-empty").classList.add("hidden");
  document.querySelector("#result-content").classList.remove("hidden");
  document.querySelector("#predicted-orders").textContent = formatNumber(result.predicted_orders, 2);
  document.querySelector("#p-low").textContent = formatNumber(result.p_low, 2);
  document.querySelector("#p-median").textContent = formatNumber(result.p_median, 2);
  document.querySelector("#p-high").textContent = formatNumber(result.p_high, 2);
  document.querySelector("#model-version").textContent = result.model_version;
  document.querySelector("#calculated-at").textContent = formatDateTime();

  const latest = request.recent_orders.at(-1);
  const difference = result.predicted_orders - latest;
  const comparison = document.querySelector("#result-comparison");
  comparison.textContent = `Son haftaya göre ${difference >= 0 ? "+" : ""}${formatNumber(difference, 0)} porsiyon`;
  comparison.style.color = difference >= 0 ? "var(--success)" : "var(--warning)";
  comparison.style.background = difference >= 0 ? "var(--success-soft)" : "var(--warning-soft)";

  const scaleMin = Math.max(0, result.p_low * 0.86);
  const scaleMax = Math.max(result.p_high * 1.08, result.predicted_orders + 1);
  const scale = (value) => Math.max(0, Math.min(100, ((value - scaleMin) / (scaleMax - scaleMin)) * 100));
  const band = document.querySelector("#range-band");
  band.style.left = `${scale(result.p_low)}%`;
  band.style.width = `${Math.max(2, scale(result.p_high) - scale(result.p_low))}%`;
  document.querySelector("#range-marker").style.left = `${scale(result.predicted_orders)}%`;

  const notice = document.querySelector("#result-notice");
  if (result.warnings?.length) {
    notice.className = "result-notice warning";
    notice.querySelector("strong").textContent = "Sipariş geçmişi yetersiz";
    notice.querySelector("p").textContent = "Dört haftadan az veri olduğu için üretim kararını işletme koşullarıyla birlikte değerlendirin.";
  } else {
    notice.className = "result-notice success";
    notice.querySelector("strong").textContent = "Son haftaların verisi yeterli";
    notice.querySelector("p").textContent = "Öneri, girdiğiniz dört haftalık gerçek sipariş geçmişine göre hazırlandı.";
  }
}

function createCell(row, value, className = "") {
  const cell = row.insertCell();
  cell.textContent = value;
  if (className) cell.className = className;
  return cell;
}

function statusPill(text, type) {
  const span = document.createElement("span");
  span.className = `table-status ${type}`;
  span.textContent = text;
  return span;
}

function renderHistory() {
  const body = document.querySelector("#history-body");
  body.replaceChildren();
  document.querySelector("#export-history").disabled = state.history.length === 0;
  if (!state.history.length) {
    const row = body.insertRow();
    row.className = "empty-row";
    const cell = row.insertCell();
    cell.colSpan = 6;
    cell.textContent = "Bu oturumda henüz tahmin yapılmadı.";
    return;
  }
  state.history.slice(0, 5).forEach(({ request, result }) => {
    const row = body.insertRow();
    createCell(row, String(request.week));
    createCell(row, String(request.center_id));
    createCell(row, String(request.meal_id));
    createCell(row, `${formatNumber(result.predicted_orders, 2)} pors.`);
    createCell(row, `${formatNumber(result.p_low, 0)} – ${formatNumber(result.p_high, 0)}`);
    const status = row.insertCell();
    status.appendChild(statusPill(result.warnings?.length ? "Uyarılı" : "Hesaplandı", result.warnings?.length ? "warning" : "success"));
  });
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  const normalized = text.replace(/^\uFEFF/, "");
  for (let index = 0; index < normalized.length; index += 1) {
    const character = normalized[index];
    if (character === '"') {
      if (quoted && normalized[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      row.push(field.trim());
      field = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && normalized[index + 1] === "\n") index += 1;
      row.push(field.trim());
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      field = "";
    } else {
      field += character;
    }
  }
  row.push(field.trim());
  if (row.some((value) => value !== "")) rows.push(row);
  if (quoted) throw new Error("CSV içinde kapanmamış tırnak işareti var.");
  return rows;
}

function parseHistory(value) {
  if (!value.trim()) return [];
  return value.split(/[|;]/).map((item) => Number(item.trim()));
}

function normalizeCsvRow(raw, rowNumber) {
  const request = {
    week: Number(raw.week),
    center_id: Number(raw.center_id),
    meal_id: Number(raw.meal_id),
    checkout_price: Number(raw.checkout_price),
    base_price: Number(raw.base_price),
    emailer_for_promotion: Number(raw.emailer_for_promotion),
    homepage_featured: Number(raw.homepage_featured),
    category: raw.category,
    cuisine: raw.cuisine,
    center_type: raw.center_type,
    city_code: Number(raw.city_code),
    region_code: Number(raw.region_code),
    op_area: Number(raw.op_area),
    recent_orders: parseHistory(raw.recent_orders),
  };
  const errors = [];
  const finiteFields = ["week", "center_id", "meal_id", "checkout_price", "base_price", "city_code", "region_code", "op_area"];
  finiteFields.forEach((name) => {
    if (!Number.isFinite(request[name])) errors.push(`${name} sayısal olmalı`);
  });
  if (request.week < 1 || request.week > 520) errors.push("week 1–520 aralığında olmalı");
  if (request.checkout_price <= 0 || request.base_price <= 0) errors.push("fiyatlar sıfırdan büyük olmalı");
  if (![0, 1].includes(request.emailer_for_promotion) || ![0, 1].includes(request.homepage_featured)) errors.push("promosyon alanları 0 veya 1 olmalı");
  if (request.recent_orders.some((value) => !Number.isFinite(value) || value < 0)) errors.push("recent_orders negatif olmayan sayılardan oluşmalı");
  if (request.recent_orders.length > 13) errors.push("recent_orders en fazla 13 değer içerebilir");

  const categories = state.modelInfo?.categories;
  if (categories) {
    if (!categories.category.includes(request.category)) errors.push("kategori model sözlüğünde yok");
    if (!categories.cuisine.includes(request.cuisine)) errors.push("mutfak model sözlüğünde yok");
    if (!categories.center_type.includes(request.center_type)) errors.push("merkez türü model sözlüğünde yok");
  }
  const warnings = request.recent_orders.length < 4 ? ["Dört haftadan az geçmiş sipariş"] : [];
  return { rowNumber, raw, request, errors, warnings, result: null };
}

async function handleCsvFile(file) {
  if (!file || !file.name.toLowerCase().endsWith(".csv")) {
    showToast("Lütfen .csv uzantılı bir dosya seçin.", "error");
    return;
  }
  try {
    const rows = parseCsv(await file.text());
    if (rows.length < 2) throw new Error("CSV başlık ve en az bir veri satırı içermeli.");
    const headers = rows[0].map((header) => header.trim());
    headers[0] = headers[0].replace(/^\uFEFF/, "");
    const missing = requiredCsvColumns.filter((column) => !headers.includes(column));
    if (missing.length) throw new Error(`Eksik sütunlar: ${missing.join(", ")}`);
    if (rows.length - 1 > 500) throw new Error("CSV en fazla 500 veri satırı içerebilir.");

    state.batchRows = rows.slice(1).map((values, index) => {
      const raw = Object.fromEntries(headers.map((header, columnIndex) => [header, values[columnIndex] ?? ""]));
      return normalizeCsvRow(raw, index + 2);
    });
    state.batchResults = [];
    document.querySelector("#file-summary").textContent = `${file.name} · ${state.batchRows.length} kayıt okundu`;
    document.querySelector("#file-summary").classList.remove("hidden");
    document.querySelector("#run-batch").disabled = !state.batchRows.some((row) => row.errors.length === 0);
    document.querySelector("#clear-batch").disabled = false;
    document.querySelector("#export-batch").disabled = true;
    renderBatchTable();
  } catch (error) {
    clearBatch();
    showToast(error.message, "error");
  }
}

function renderBatchSummary() {
  const valid = state.batchRows.filter((row) => row.errors.length === 0 && row.warnings.length === 0).length;
  const warning = state.batchRows.filter((row) => row.errors.length === 0 && row.warnings.length > 0).length;
  const invalid = state.batchRows.filter((row) => row.errors.length > 0).length;
  document.querySelector("#total-row-count").textContent = String(state.batchRows.length);
  document.querySelector("#valid-row-count").textContent = String(valid);
  document.querySelector("#warning-row-count").textContent = String(warning);
  document.querySelector("#invalid-row-count").textContent = String(invalid);
}

function renderBatchTable() {
  renderBatchSummary();
  const body = document.querySelector("#batch-body");
  body.replaceChildren();
  if (!state.batchRows.length) {
    const row = body.insertRow();
    row.className = "empty-row";
    const cell = row.insertCell();
    cell.colSpan = 8;
    cell.textContent = "Henüz CSV dosyası seçilmedi.";
    document.querySelector("#batch-table-description").textContent = "Dosya seçildiğinde doğrulama sonucu burada gösterilir.";
    return;
  }
  document.querySelector("#batch-table-description").textContent = `${state.batchRows.length} kayıt doğrulandı; ilk 100 kayıt gösteriliyor.`;
  state.batchRows.slice(0, 100).forEach((item) => {
    const row = body.insertRow();
    createCell(row, String(item.rowNumber));
    createCell(row, String(item.request.week || "—"));
    createCell(row, String(item.request.center_id ?? "—"));
    createCell(row, String(item.request.meal_id ?? "—"));
    createCell(row, displayLabels.category[item.request.category] || item.request.category || "—");
    createCell(row, item.result ? `${formatNumber(item.result.predicted_orders, 2)} pors.` : "—");
    createCell(row, item.result ? `${formatNumber(item.result.p_low, 0)} – ${formatNumber(item.result.p_high, 0)}` : "—");
    const status = row.insertCell();
    if (item.errors.length) status.appendChild(statusPill(item.errors.join("; "), "error"));
    else if (item.result) status.appendChild(statusPill(item.result.warnings?.length ? "Uyarılı" : "Hesaplandı", item.result.warnings?.length ? "warning" : "success"));
    else if (item.warnings.length) status.appendChild(statusPill(item.warnings.join("; "), "warning"));
    else status.appendChild(statusPill("Geçerli", "success"));
  });
}

async function runBatch() {
  const validRows = state.batchRows.filter((row) => row.errors.length === 0);
  if (!validRows.length) return;
  const button = document.querySelector("#run-batch");
  button.disabled = true;
  button.textContent = "Hesaplanıyor…";
  try {
    const response = await apiFetch("/predict/batch", {
      method: "POST",
      body: JSON.stringify(validRows.map((row) => row.request)),
    });
    const payload = await response.json();
    validRows.forEach((row, index) => { row.result = payload.results[index]; });
    state.batchResults = validRows;
    document.querySelector("#export-batch").disabled = false;
    renderBatchTable();
    showToast(`${payload.count} yemek için üretim önerisi hazırlandı.`);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Kontrol et ve planı oluştur";
  }
}

function clearBatch() {
  state.batchRows = [];
  state.batchResults = [];
  document.querySelector("#batch-file").value = "";
  document.querySelector("#file-summary").classList.add("hidden");
  document.querySelector("#run-batch").disabled = true;
  document.querySelector("#clear-batch").disabled = true;
  document.querySelector("#export-batch").disabled = true;
  renderBatchTable();
}

function safeCsvValue(value) {
  let text = String(value ?? "");
  if (/^[=+@-]/.test(text)) text = `'${text}`;
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadCsv(filename, headers, rows) {
  const csv = [headers, ...rows].map((row) => row.map(safeCsvValue).join(",")).join("\r\n");
  const blob = new Blob(["\uFEFF", csv], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function downloadSampleCsv() {
  downloadCsv("toplu_tahmin_ornek.csv", requiredCsvColumns, [
    [136, 10, 1062, 194.06, 194.06, 0, 0, "Beverages", "Italian", "TYPE_B", 590, 56, 6.3, "1538|700|704|960"],
    [137, 10, 1062, 198.00, 200.00, 1, 0, "Beverages", "Italian", "TYPE_B", 590, 56, 6.3, "700|704|960|921"],
  ]);
}

function exportHistory() {
  downloadCsv("oturum_tahminleri.csv",
    ["created_at", ...requiredCsvColumns.filter((column) => column !== "recent_orders"), "recent_orders", "predicted_orders", "p_low", "p_median", "p_high", "model_version", "warnings"],
    state.history.map(({ createdAt, request, result }) => [
      createdAt, request.week, request.center_id, request.meal_id, request.checkout_price, request.base_price,
      request.emailer_for_promotion, request.homepage_featured, request.category, request.cuisine, request.center_type,
      request.city_code, request.region_code, request.op_area, request.recent_orders.join("|"), result.predicted_orders,
      result.p_low, result.p_median, result.p_high, result.model_version, result.warnings.join("|"),
    ]));
}

function exportBatch() {
  downloadCsv("toplu_tahmin_sonuclari.csv",
    ["source_row", "week", "center_id", "meal_id", "category", "predicted_orders", "p_low", "p_median", "p_high", "model_version", "warnings"],
    state.batchResults.map(({ rowNumber, request, result }) => [
      rowNumber, request.week, request.center_id, request.meal_id, request.category, result.predicted_orders,
      result.p_low, result.p_median, result.p_high, result.model_version, result.warnings.join("|"),
    ]));
}

function renderValidationMetrics() {
  if (!state.modelInfo) return;
  const metrics = state.modelInfo.reported_metrics;
  document.querySelector("#holdout-rmsle").textContent = formatNumber(metrics.holdout_rmsle, 4);
  document.querySelector("#walk-rmsle").textContent = formatNumber(metrics.walk_forward_rmsle, 4);
  document.querySelector("#persistence-rmsle").textContent = formatNumber(metrics.persistence_rmsle, 4);
  document.querySelector("#band-coverage").textContent = formatPercent(metrics.band_coverage, 2);

  const chart = document.querySelector("#validation-chart");
  chart.replaceChildren();
  const rows = [
    ["Son dönem modeli", metrics.holdout_rmsle, false],
    ["Üç dönem ortalaması", metrics.walk_forward_rmsle, false],
    ["Sadece geçen haftayı kullanan basit yöntem", metrics.persistence_rmsle, true],
  ];
  const maximum = Math.max(...rows.map((row) => row[1])) * 1.08;
  rows.forEach(([label, value, reference]) => {
    const row = document.createElement("div");
    row.className = `bar-row ${reference ? "reference" : ""}`;
    const name = document.createElement("span");
    name.textContent = label;
    const track = document.createElement("div");
    track.className = "bar-track";
    const fill = document.createElement("div");
    fill.className = "bar-fill";
    fill.style.width = `${(value / maximum) * 100}%`;
    track.appendChild(fill);
    const number = document.createElement("strong");
    number.textContent = formatNumber(value, 4);
    row.append(name, track, number);
    chart.appendChild(row);
  });
}

function sumPrometheusMetric(text, name) {
  return text.split(/\r?\n/).reduce((sum, line) => {
    if (!line.startsWith(name) || line.startsWith(`${name}_created`)) return sum;
    const match = line.match(/\s(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)$/i);
    return sum + (match ? Number(match[1]) : 0);
  }, 0);
}

async function loadLiveMetrics() {
  if (!state.apiKey) return;
  const note = document.querySelector("#monitoring-note");
  try {
    const response = await apiFetch("/metrics", { cache: "no-store" });
    const text = await response.text();
    document.querySelector("#request-total").textContent = formatNumber(sumPrometheusMetric(text, "api_requests_total"));
    document.querySelector("#cold-start-total").textContent = formatNumber(sumPrometheusMetric(text, "prediction_cold_start_total"));
    document.querySelector("#auth-failure-total").textContent = formatNumber(sumPrometheusMetric(text, "auth_failures_total"));
    document.querySelector("#rate-limited-total").textContent = formatNumber(sumPrometheusMetric(text, "rate_limited_total"));
    note.textContent = `Son yenileme: ${formatDateTime()}`;
  } catch (error) {
    note.textContent = `Canlı metrikler yüklenemedi: ${error.message}`;
  }
}

async function refreshMonitoring() {
  if (!state.apiKey) {
    renderSessionChart();
    return;
  }
  try {
    if (!state.modelInfo) await loadModelInfo();
    await loadLiveMetrics();
  } catch (error) {
    showToast(error.message, "error");
  }
  renderSessionChart();
}

function renderSessionChart() {
  const container = document.querySelector("#session-chart");
  const historyItems = [...state.history].reverse();
  document.querySelector("#session-count").textContent = `${historyItems.length} tahmin`;
  if (historyItems.length < 2) {
    container.replaceChildren();
    const message = document.createElement("p");
    message.className = "empty-chart";
    message.textContent = "Grafik için en az iki tahmin oluşturun.";
    container.appendChild(message);
    return;
  }

  const width = 860;
  const height = 270;
  const margin = { left: 56, right: 24, top: 24, bottom: 42 };
  const values = historyItems.flatMap((item) => [item.result.p_low, item.result.p_high, item.result.predicted_orders]);
  const minValue = Math.max(0, Math.min(...values) * 0.88);
  const maxValue = Math.max(...values) * 1.08;
  const x = (index) => margin.left + (index / (historyItems.length - 1)) * (width - margin.left - margin.right);
  const y = (value) => margin.top + ((maxValue - value) / (maxValue - minValue || 1)) * (height - margin.top - margin.bottom);
  const pointString = (selector) => historyItems.map((item, index) => `${x(index)},${y(selector(item.result))}`).join(" ");
  const band = [
    ...historyItems.map((item, index) => `${x(index)},${y(item.result.p_high)}`),
    ...historyItems.map((item, index) => `${x(historyItems.length - 1 - index)},${y(historyItems[historyItems.length - 1 - index].result.p_low)}`),
  ].join(" ");

  const axisLabels = [0, 0.5, 1].map((ratio) => {
    const value = maxValue - (maxValue - minValue) * ratio;
    const yPosition = margin.top + ratio * (height - margin.top - margin.bottom);
    return `<line x1="${margin.left}" y1="${yPosition}" x2="${width - margin.right}" y2="${yPosition}" stroke="#e2e7e4"/><text x="${margin.left - 8}" y="${yPosition + 4}" text-anchor="end" fill="#6a746f" font-size="11">${formatNumber(value, 0)}</text>`;
  }).join("");
  const xLabels = historyItems.map((item, index) => `<text x="${x(index)}" y="${height - 14}" text-anchor="middle" fill="#6a746f" font-size="11">H${item.request.week}</text>`).join("");

  container.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
      ${axisLabels}
      <polygon points="${band}" fill="#d8eee5" opacity="0.9"></polygon>
      <polyline points="${pointString((result) => result.predicted_orders)}" fill="none" stroke="#174f3c" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></polyline>
      ${historyItems.map((item, index) => `<circle cx="${x(index)}" cy="${y(item.result.predicted_orders)}" r="4" fill="#ffffff" stroke="#174f3c" stroke-width="2"><title>H${item.request.week}: ${formatNumber(item.result.predicted_orders, 2)} porsiyon</title></circle>`).join("")}
      ${xLabels}
    </svg>`;
}

function attachEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => button.addEventListener("click", () => setView(button.dataset.view)));
  document.querySelectorAll("[data-view-link]").forEach((button) => button.addEventListener("click", () => setView(button.dataset.viewLink)));
  document.querySelector("#open-api-settings").addEventListener("click", () => openApiDialog());
  document.querySelector("#banner-api-settings").addEventListener("click", () => openApiDialog());
  document.querySelector("#save-api-key").addEventListener("click", saveApiKey);
  document.querySelector("#prediction-form").addEventListener("submit", submitPrediction);
  document.querySelector("#quick-form").addEventListener("submit", submitQuickPrediction);
  document.querySelector("#quick-example").addEventListener("click", fillQuickExample);
  document.querySelector("#export-history").addEventListener("click", exportHistory);
  document.querySelector("#download-sample").addEventListener("click", downloadSampleCsv);
  document.querySelector("#run-batch").addEventListener("click", runBatch);
  document.querySelector("#clear-batch").addEventListener("click", clearBatch);
  document.querySelector("#export-batch").addEventListener("click", exportBatch);
  document.querySelector("#refresh-monitoring").addEventListener("click", refreshMonitoring);

  const fileInput = document.querySelector("#batch-file");
  const dropZone = document.querySelector("#drop-zone");
  fileInput.addEventListener("change", () => handleCsvFile(fileInput.files[0]));
  ["dragenter", "dragover"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  }));
  dropZone.addEventListener("drop", (event) => handleCsvFile(event.dataTransfer.files[0]));
  window.addEventListener("hashchange", () => setView(viewFromHash()));
}

async function initialize() {
  attachEvents();
  updateConnectionBanner();
  renderHistory();
  renderQuickHistory();
  renderSessionChart();
  setView(viewFromHash());
  await checkHealth();
  if (state.apiKey) {
    try {
      await loadModelInfo();
      await loadLiveMetrics();
    } catch {
      sessionStorage.removeItem(API_KEY_STORAGE);
      state.apiKey = "";
      updateConnectionBanner();
    }
  }
}

initialize();
