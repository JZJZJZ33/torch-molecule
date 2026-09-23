const input = document.querySelector("#smiles-input");
const predictButton = document.querySelector("#predict-button");
const errorMessage = document.querySelector("#error-message");
const structureCanvas = document.querySelector("#structure-canvas");
const structure3d = document.querySelector("#structure-3d");
const structure2d = document.querySelector("#structure-2d");
const conformerStatus = document.querySelector("#conformer-status");
const toggleHydrogensButton = document.querySelector("#toggle-hydrogens");
const resetViewButton = document.querySelector("#reset-view");
const propertyGrid = document.querySelector("#property-grid");
const resultsSummary = document.querySelector("#results-summary");
const serviceStatus = document.querySelector("#service-status");
const statusText = document.querySelector("#status-text");
const isChinese = document.documentElement.lang.startsWith("zh");
const ui = (english, chinese) => (isChinese ? chinese : english);

let molecularViewer = null;
let hydrogensVisible = false;

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
    statusText.textContent = ui(`${health.property_count} models ready`, `${health.property_count} 个模型已就绪`);
  } catch (error) {
    serviceStatus.classList.add("error");
    statusText.textContent = ui("Models unavailable", "模型不可用");
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
  conformerStatus.textContent = ui("Generating 10-unit chain…", "正在生成 10 单元链…");
  toggleHydrogensButton.disabled = true;
  resetViewButton.disabled = true;
}

function applyViewerStyle() {
  molecularViewer.setStyle({}, {
    stick: { radius: 0.14, colorscheme: "Jmol" },
    sphere: { scale: 0.23, colorscheme: "Jmol" },
  });
  if (!hydrogensVisible) molecularViewer.setStyle({ elem: "H" }, {});
  molecularViewer.render();
}

function renderStructure(structure) {
  if (typeof window.$3Dmol === "undefined") {
    throw new Error(ui("The 3D viewer library could not be loaded.", "无法加载三维查看器。"));
  }

  structure2d.innerHTML = structure.svg_2d;
  if (molecularViewer === null) {
    structure3d.replaceChildren();
    molecularViewer = window.$3Dmol.createViewer(structure3d, {
      backgroundColor: "#f8fbfe",
      antialias: true,
    });
  } else {
    molecularViewer.clear();
  }

  molecularViewer.addModel(`${structure.mol_block}\n$$$$\n`, "sdf");
  hydrogensVisible = false;
  toggleHydrogensButton.textContent = ui("Show hydrogens", "显示氢原子");
  applyViewerStyle();
  molecularViewer.zoomTo();
  molecularViewer.render();
  molecularViewer.resize();

  const convergence = structure.uff_converged ? ui("converged", "已收敛") : ui("iteration limit reached", "达到迭代上限");
  conformerStatus.textContent = ui(`${structure.repeat_units} units · optimization ${convergence} · ${structure.uff_energy.toFixed(2)} kcal/mol`, `${structure.repeat_units} 个单元 · 优化${convergence} · ${structure.uff_energy.toFixed(2)} kcal/mol`);
  toggleHydrogensButton.disabled = false;
  resetViewButton.disabled = false;
}

function showStructureError(message) {
  if (molecularViewer !== null) {
    molecularViewer.clear();
    molecularViewer.render();
  }
  structure3d.innerHTML = `<div class="empty-state"><p>${ui("Unable to generate 3D conformer.", "无法生成三维构象。")}</p></div>`;
  structure2d.innerHTML = `<div class="empty-state"><p>${ui("2D structure unavailable.", "二维结构不可用。")}</p></div>`;
  molecularViewer = null;
  conformerStatus.textContent = message;
  toggleHydrogensButton.disabled = true;
  resetViewButton.disabled = true;
}

function renderProperties(predictions) {
  propertyGrid.replaceChildren();
  const entries = Object.entries(predictions);
  entries.forEach(([name, values], index) => {
    const row = document.createElement("tr");
    row.style.animationDelay = `${Math.min(index * 25, 250)}ms`;
    const english = document.createElement("td");
    english.className = "property-english";
    english.textContent = isChinese ? (propertyLabelsZh[name] || name.replaceAll("_", " ")) : (propertyLabels[name] || name.replaceAll("_", " "));
    const chinese = document.createElement("td");
    chinese.className = "property-chinese";
    chinese.textContent = isChinese ? (propertyLabels[name] || name.replaceAll("_", " ")) : (propertyLabelsZh[name] || "属性预测");
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
  resultsSummary.textContent = ui(`${entries.length} properties predicted from one polymer structure.`, `已从一个聚合物结构预测 ${entries.length} 项性能。`);
}

async function predict() {
  const smiles = input.value.trim();
  if (!smiles) {
    errorMessage.textContent = ui("Enter a pSMILES string first.", "请先输入 pSMILES 字符串。");
    input.focus();
    return;
  }

  errorMessage.textContent = "";
  predictButton.disabled = true;
  predictButton.querySelector("span:first-child").textContent = ui("Running property models…", "正在运行性能模型…");
  showLoading();

  const structureRequest = fetch("./structure", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles }),
      }).then(async (response) => {
        if (!response.ok) throw new Error(await parseError(response));
        return response.json();
      });
  const predictionRequest = fetch("./predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ smiles: [smiles] }),
      }).then(async (response) => {
        if (!response.ok) throw new Error(await parseError(response));
        return response.json();
      });

  try {
    const [structureResult, predictionResult] = await Promise.allSettled([
      structureRequest,
      predictionRequest,
    ]);
    const errors = [];

    if (structureResult.status === "fulfilled") {
      try {
        renderStructure(structureResult.value);
      } catch (error) {
        showStructureError(error.message || "Viewer failed");
        errors.push(`Structure: ${error.message || "viewer failed"}`);
      }
    } else {
      const message = structureResult.reason?.message || "conformer generation failed";
      showStructureError(message);
      errors.push(`Structure: ${message}`);
    }

    if (predictionResult.status === "fulfilled") {
      renderProperties(predictionResult.value.predictions);
    } else {
      propertyGrid.innerHTML = `<tr class="results-placeholder"><td colspan="5">${ui("Prediction unavailable", "预测不可用")}</td></tr>`;
      resultsSummary.textContent = ui("Property prediction failed; the structure may still be explored.", "性能预测失败，但仍可查看结构。");
      errors.push(`Prediction: ${predictionResult.reason?.message || "request failed"}`);
    }

    errorMessage.textContent = errors.join(" · ");
  } finally {
    structureCanvas.classList.remove("loading-shimmer");
    predictButton.disabled = false;
    predictButton.querySelector("span:first-child").textContent = ui("Predict all properties", "预测全部性能");
  }
}

toggleHydrogensButton.addEventListener("click", () => {
  if (molecularViewer === null) return;
  hydrogensVisible = !hydrogensVisible;
  toggleHydrogensButton.textContent = hydrogensVisible ? ui("Hide hydrogens", "隐藏氢原子") : ui("Show hydrogens", "显示氢原子");
  applyViewerStyle();
});

resetViewButton.addEventListener("click", () => {
  if (molecularViewer === null) return;
  molecularViewer.zoomTo();
  molecularViewer.render();
});

window.addEventListener("resize", () => {
  if (molecularViewer !== null) molecularViewer.resize();
});

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
