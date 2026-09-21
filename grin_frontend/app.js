const input = document.querySelector("#smiles-input");
const predictButton = document.querySelector("#predict-button");
const errorMessage = document.querySelector("#error-message");
const structureCanvas = document.querySelector("#structure-canvas");
const propertyGrid = document.querySelector("#property-grid");
const resultsSummary = document.querySelector("#results-summary");
const serviceStatus = document.querySelector("#service-status");
const statusText = document.querySelector("#status-text");

const propertyLabels = {
  density: "Density",
  Rg: "Radius of gyration",
  "self-diffusion": "Self-diffusion",
  cp: "Constant-pressure heat capacity",
  Cv: "Constant-volume heat capacity",
  compressibility: "Compressibility",
  isentropic_compressibility: "Isentropic compressibility",
  bulk_modulus: "Bulk modulus",
  isentropic_bulk_modulus: "Isentropic bulk modulus",
  volume_expansion: "Volume expansion coefficient",
  linear_expansion: "Linear expansion coefficient",
  r2: "Mean square end-to-end distance",
  static_dielectric_const: "Static dielectric constant",
  nematic_order_parameter: "Nematic order parameter",
  refractive_index: "Refractive index",
  thermal_conductivity: "Thermal conductivity",
  thermal_diffusivity: "Thermal diffusivity",
  tg: "Glass-transition temperature",
};

const propertyLabelsZh = {
  density: "密度",
  Rg: "回转半径",
  "self-diffusion": "自扩散系数",
  cp: "定压比热容",
  Cv: "定容比热容",
  compressibility: "压缩系数",
  isentropic_compressibility: "等熵压缩系数",
  bulk_modulus: "体积模量",
  isentropic_bulk_modulus: "等熵体积模量",
  volume_expansion: "体积膨胀系数",
  linear_expansion: "线性膨胀系数",
  r2: "均方末端距",
  static_dielectric_const: "静态介电常数",
  nematic_order_parameter: "向列序参数",
  refractive_index: "折射率",
  thermal_conductivity: "热导率",
  thermal_diffusivity: "热扩散率",
  tg: "玻璃化转变温度",
};

const propertyUnits = {
  density: "g/cm³",
  Rg: "Å",
  "self-diffusion": "m²/s",
  cp: "J/(kg·K)",
  Cv: "J/(kg·K)",
  compressibility: "Pa⁻¹",
  isentropic_compressibility: "Pa⁻¹",
  bulk_modulus: "Pa",
  isentropic_bulk_modulus: "Pa",
  volume_expansion: "K⁻¹",
  linear_expansion: "K⁻¹",
  r2: "Å",
  static_dielectric_const: "dimensionless",
  nematic_order_parameter: "dimensionless",
  refractive_index: "dimensionless",
  thermal_conductivity: "W/(m·K)",
  thermal_diffusivity: "m²/s",
  tg: "K",
};

const propertySymbols = {
  density: "ρ",
  Rg: "Rᵍ",
  "self-diffusion": "D",
  cp: "Cₚ",
  Cv: "Cᵥ",
  compressibility: "κ",
  isentropic_compressibility: "κₛ",
  bulk_modulus: "K",
  isentropic_bulk_modulus: "Kₛ",
  volume_expansion: "αᵥ",
  linear_expansion: "αₗ",
  r2: "R²",
  static_dielectric_const: "ε",
  nematic_order_parameter: "S",
  refractive_index: "n",
  thermal_conductivity: "λ",
  thermal_diffusivity: "a",
  tg: "Tg",
};

function formatValue(value) {
  const absolute = Math.abs(value);
  if ((absolute !== 0 && absolute < 0.001) || absolute >= 100000) {
    const exponent = Math.floor(Math.log10(absolute));
    const coefficient = value / (10 ** exponent);
    return {
      coefficient: new Intl.NumberFormat("en-US", { maximumSignificantDigits: 6 }).format(coefficient),
      exponent,
    };
  }
  return { plain: new Intl.NumberFormat("en-US", { maximumSignificantDigits: 7 }).format(value) };
}

function appendFormattedValue(container, value) {
  const formatted = formatValue(value);
  if (formatted.plain !== undefined) {
    container.textContent = formatted.plain;
    return;
  }
  container.append(document.createTextNode(`${formatted.coefficient} × 10`));
  const exponent = document.createElement("sup");
  exponent.textContent = String(formatted.exponent).replace("-", "−");
  container.append(exponent);
}

async function parseError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") return payload.detail;
    return payload.detail?.message || JSON.stringify(payload.detail);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

async function checkHealth() {
  try {
    const response = await fetch("./health");
    if (!response.ok) throw new Error(await parseError(response));
    const health = await response.json();
    serviceStatus.classList.add("ready");
    statusText.textContent = `${health.property_count} models ready`;
  } catch (error) {
    serviceStatus.classList.add("error");
    statusText.textContent = "Models unavailable";
  }
}

function showLoading() {
  propertyGrid.replaceChildren();
  for (let index = 0; index < 5; index += 1) {
    const row = document.createElement("tr");
    row.className = "loading-row";
    for (let cellIndex = 0; cellIndex < 5; cellIndex += 1) {
      const cell = document.createElement("td");
      const bar = document.createElement("span");
      bar.className = "loading-bar loading-shimmer";
      cell.append(bar);
      row.append(cell);
    }
    propertyGrid.append(row);
  }
  structureCanvas.classList.add("loading-shimmer");
}

function renderProperties(predictions) {
  propertyGrid.replaceChildren();
  const entries = Object.entries(predictions);
  entries.forEach(([name, values], index) => {
    const row = document.createElement("tr");
    row.style.animationDelay = `${Math.min(index * 25, 250)}ms`;
    const english = document.createElement("td");
    english.className = "property-english";
    english.textContent = propertyLabels[name] || name.replaceAll("_", " ");
    const chinese = document.createElement("td");
    chinese.className = "property-chinese";
    chinese.textContent = propertyLabelsZh[name] || "属性预测";
    const symbol = document.createElement("td");
    symbol.className = "property-symbol-cell";
    symbol.textContent = propertySymbols[name] || "—";
    const value = document.createElement("td");
    value.className = "property-value-cell numeric-column";
    appendFormattedValue(value, values[0]);
    const unit = document.createElement("td");
    unit.className = "property-unit-cell";
    unit.textContent = propertyUnits[name] || "RPPD unit";
    row.append(english, chinese, symbol, value, unit);
    propertyGrid.append(row);
  });
  resultsSummary.textContent = `${entries.length} properties predicted from one polymer structure.`;
}

async function predict() {
  const smiles = input.value.trim();
  if (!smiles) {
    errorMessage.textContent = "Enter a pSMILES string first.";
    input.focus();
    return;
  }

  errorMessage.textContent = "";
  predictButton.disabled = true;
  predictButton.querySelector("span:first-child").textContent = "Running GRIN models…";
  showLoading();

  try {
    const [structureResponse, predictionResponse] = await Promise.all([
      fetch("./structure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      }),
      fetch("./predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: [smiles] }),
      }),
    ]);

    if (!structureResponse.ok) throw new Error(await parseError(structureResponse));
    if (!predictionResponse.ok) throw new Error(await parseError(predictionResponse));

    const svg = await structureResponse.text();
    const prediction = await predictionResponse.json();
    structureCanvas.classList.remove("loading-shimmer");
    structureCanvas.innerHTML = svg;
    renderProperties(prediction.predictions);
  } catch (error) {
    structureCanvas.classList.remove("loading-shimmer");
    structureCanvas.innerHTML = `<div class="empty-state"><p>Unable to render structure.</p></div>`;
    propertyGrid.innerHTML = `<tr class="results-placeholder"><td colspan="5">Prediction unavailable</td></tr>`;
    errorMessage.textContent = error.message || "Prediction failed.";
    resultsSummary.textContent = "Check the pSMILES and API model configuration.";
  } finally {
    predictButton.disabled = false;
    predictButton.querySelector("span:first-child").textContent = "Predict all properties";
  }
}

predictButton.addEventListener("click", predict);
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter") predict();
});
document.querySelectorAll(".example-button").forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.smiles;
    input.focus();
  });
});
document.querySelectorAll('.header-nav [aria-disabled="true"]').forEach((link) => {
  link.addEventListener("click", (event) => event.preventDefault());
});

checkHealth();
