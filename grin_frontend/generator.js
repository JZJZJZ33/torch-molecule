const propertyOptions = document.querySelector("#generation-property-options");
const conditionInputs = document.querySelector("#generation-condition-inputs");
const generationCount = document.querySelector("#generation-count");
const generateButton = document.querySelector("#generate-button");
const modelStatus = document.querySelector("#generation-model-status");
const errorMessage = document.querySelector("#generation-error");
const results = document.querySelector("#generation-results");
const resultsSummary = document.querySelector("#generation-results-summary");
const serviceStatus = document.querySelector("#service-status");
const statusText = document.querySelector("#status-text");
const isChinese = document.documentElement.lang.startsWith("zh");
const ui = (english, chinese) => (isChinese ? chinese : english);
const apiPrefix = isChinese ? "../" : "./";

let models = new Map();

const labels = {
  density: "Density", Rg: "Radius of gyration", "self-diffusion": "Self-diffusion",
  cp: "Constant-pressure heat capacity", Cv: "Constant-volume heat capacity",
  compressibility: "Compressibility", isentropic_compressibility: "Isentropic compressibility",
  bulk_modulus: "Bulk modulus", isentropic_bulk_modulus: "Isentropic bulk modulus",
  volume_expansion: "Volume expansion coefficient", linear_expansion: "Linear expansion coefficient",
  static_dielectric_const: "Static dielectric constant", nematic_order_parameter: "Nematic order parameter",
  refractive_index: "Refractive index", thermal_conductivity: "Thermal conductivity",
  thermal_diffusivity: "Thermal diffusivity", tg: "Glass-transition temperature",
};

const labelsZh = {
  density: "密度", Rg: "回转半径", "self-diffusion": "自扩散系数",
  cp: "定压比热容", Cv: "定容比热容", compressibility: "压缩系数",
  isentropic_compressibility: "等熵压缩系数", bulk_modulus: "体积模量",
  isentropic_bulk_modulus: "等熵体积模量", volume_expansion: "体积膨胀系数",
  linear_expansion: "线性膨胀系数", static_dielectric_const: "静态介电常数",
  nematic_order_parameter: "向列序参数", refractive_index: "折射率",
  thermal_conductivity: "热导率", thermal_diffusivity: "热扩散率", tg: "玻璃化转变温度",
};

function propertyLabel(name) {
  return (isChinese ? labelsZh[name] : labels[name]) || name.replaceAll("_", " ");
}

const units = {
  density: "g/cm³", Rg: "Å", "self-diffusion": "m²/s", cp: "J/(kg·K)", Cv: "J/(kg·K)",
  compressibility: "Pa⁻¹", isentropic_compressibility: "Pa⁻¹", bulk_modulus: "Pa",
  isentropic_bulk_modulus: "Pa", volume_expansion: "K⁻¹", linear_expansion: "K⁻¹",
  static_dielectric_const: "dimensionless", nematic_order_parameter: "dimensionless",
  refractive_index: "dimensionless", thermal_conductivity: "W/(m·K)",
  thermal_diffusivity: "m²/s", tg: "K",
};

async function parseError(response) {
  try {
    const payload = await response.json();
    if (typeof payload.detail === "string") return payload.detail;
    return payload.detail?.message || JSON.stringify(payload.detail);
  } catch {
    return `${response.status} ${response.statusText}`;
  }
}

function key(properties) {
  return [...properties].sort().join("|");
}

function selectedProperties() {
  return [...propertyOptions.querySelectorAll('input[data-kind="property"]:checked')]
    .map((item) => item.value);
}

function unconditionalSelected() {
  return Boolean(propertyOptions.querySelector('input[data-kind="unconditional"]:checked'));
}

function updateSelection() {
  const selected = selectedProperties();
  const unconditional = unconditionalSelected();
  const model = unconditional ? models.get("") : (selected.length ? models.get(key(selected)) : null);
  const previous = Object.fromEntries(
    [...conditionInputs.querySelectorAll("input")].map((field) => [field.dataset.property, field.value]),
  );
  conditionInputs.replaceChildren();
  selected.forEach((name) => {
    const label = document.createElement("label");
    label.className = "generation-condition";
    label.textContent = `${propertyLabel(name)} (${units[name] || ui("dataset unit", "数据单位")})`;
    const input = document.createElement("input");
    input.type = "number";
    input.step = "any";
    input.required = true;
    input.dataset.property = name;
    input.value = previous[name] || "";
    input.placeholder = ui(`Target ${name}`, `${propertyLabel(name)}目标值`);
    label.append(input);
    conditionInputs.append(label);
  });
  generateButton.disabled = !model;
  if (unconditional && model) modelStatus.textContent = ui("Using the base generator without property targets.", "使用不含性能目标的基础生成模型。");
  else if (!selected.length) modelStatus.textContent = ui("Choose no conditions or one or more properties.", "请选择无条件生成或一个及以上性能。");
  else if (model) modelStatus.textContent = ui(`Matched model: ${model.properties.map(propertyLabel).join(" + ")}`, `已匹配模型：${model.properties.map(propertyLabel).join(" + ")}`);
  else modelStatus.textContent = ui("No fine-tuned model matches this exact property combination.", "没有与此性能组合完全匹配的微调模型。");
}

async function loadModels() {
  try {
    const [healthResponse, modelResponse] = await Promise.all([
      fetch(`${apiPrefix}health`), fetch(`${apiPrefix}generation-models`),
    ]);
    if (!healthResponse.ok) throw new Error(await parseError(healthResponse));
    if (!modelResponse.ok) throw new Error(await parseError(modelResponse));
    const health = await healthResponse.json();
    const available = await modelResponse.json();
    serviceStatus.classList.add("ready");
    statusText.textContent = ui(`${health.generation_model_count} generators ready`, `${health.generation_model_count} 个生成模型已就绪`);
    models = new Map(available.map((model) => [key(model.properties), model]));
    const properties = [...new Set(available.flatMap((model) => model.properties))].sort();
    propertyOptions.replaceChildren();
    if (models.has("")) {
      const label = document.createElement("label");
      label.className = "generation-property-option generation-mode-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.dataset.kind = "unconditional";
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) {
          propertyOptions.querySelectorAll('input[data-kind="property"]').forEach((item) => { item.checked = false; });
        }
        updateSelection();
      });
      const text = document.createElement("span");
      text.textContent = ui("No conditions", "无条件生成");
      label.append(checkbox, text);
      propertyOptions.append(label);
    }
    if (!properties.length && !models.has("")) {
      propertyOptions.innerHTML = `<span class="generator-placeholder">${ui("No generation models are configured.", "尚未配置生成模型。")}</span>`;
      modelStatus.textContent = ui("Add a trained model when starting the service.", "启动服务时请添加已训练模型。");
      return;
    }
    properties.forEach((name) => {
      const label = document.createElement("label");
      label.className = "generation-property-option";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.value = name;
      checkbox.dataset.kind = "property";
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) {
          const unconditional = propertyOptions.querySelector('input[data-kind="unconditional"]');
          if (unconditional) unconditional.checked = false;
        }
        updateSelection();
      });
      const text = document.createElement("span");
      text.textContent = propertyLabel(name);
      label.append(checkbox, text);
      propertyOptions.append(label);
    });
    updateSelection();
  } catch (error) {
    serviceStatus.classList.add("error");
    statusText.textContent = ui("Generators unavailable", "生成模型不可用");
    propertyOptions.innerHTML = `<span class="generator-placeholder">${ui("Generation models unavailable.", "生成模型不可用。")}</span>`;
    errorMessage.textContent = error.message || ui("Unable to load generation models.", "无法加载生成模型。");
  }
}

function renderSamples(payload) {
  results.replaceChildren();
  const validCount = payload.samples.filter((sample) => sample.polymer_valid).length;
  resultsSummary.textContent = ui(`${validCount} of ${payload.samples.length} candidates passed repeat-unit validation.`, `${payload.samples.length} 个候选结构中有 ${validCount} 个通过重复单元验证。`);
  payload.samples.forEach((sample, index) => {
    const card = document.createElement("article");
    card.className = `generation-card${sample.polymer_valid ? " valid" : " invalid"}`;
    const heading = document.createElement("strong");
    heading.textContent = ui(`Candidate ${index + 1}`, `候选结构 ${index + 1}`);
    const status = document.createElement("span");
    status.className = "generation-card-status";
    status.textContent = sample.polymer_valid ? ui("Valid repeat unit", "有效重复单元") : sample.rejection_reason.replaceAll("_", " ");
    const smiles = document.createElement("code");
    smiles.textContent = sample.smiles || ui("No valid SMILES produced", "未生成有效 SMILES");
    card.append(heading, status, smiles);
    results.append(card);
  });
}

async function generate() {
  const properties = selectedProperties();
  const unconditional = unconditionalSelected();
  const modelKey = unconditional ? "" : key(properties);
  const conditions = {};
  for (const field of conditionInputs.querySelectorAll("input")) {
    if (field.value.trim() === "" || !Number.isFinite(Number(field.value))) {
      errorMessage.textContent = ui("Enter a finite target value for every selected property.", "请为每个所选性能输入有限目标值。");
      field.focus();
      return;
    }
    conditions[field.dataset.property] = Number(field.value);
  }
  if (!models.has(modelKey)) {
    errorMessage.textContent = ui("No model matches the selected property combination.", "没有与所选性能组合匹配的模型。");
    return;
  }
  const count = Number(generationCount.value);
  if (!Number.isInteger(count) || count < 1 || count > 100) {
    errorMessage.textContent = ui("Candidate count must be an integer from 1 to 100.", "候选数量必须是 1 至 100 的整数。");
    generationCount.focus();
    return;
  }
  errorMessage.textContent = "";
  generateButton.disabled = true;
  generateButton.querySelector("span:first-child").textContent = ui("Generating…", "正在生成…");
  results.innerHTML = `<div class="results-placeholder loading-shimmer">${ui("Generating candidates", "正在生成候选结构")}</div>`;
  resultsSummary.textContent = ui("Sampling candidates…", "正在采样候选结构…");
  try {
    const response = await fetch(`${apiPrefix}generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ conditions, number: count, batch_size: Math.min(16, count) }),
    });
    if (!response.ok) throw new Error(await parseError(response));
    renderSamples(await response.json());
  } catch (error) {
    results.innerHTML = `<div class="results-placeholder">${ui("Generation unavailable", "生成功能不可用")}</div>`;
    resultsSummary.textContent = ui("Generation failed.", "生成失败。");
    errorMessage.textContent = error.message || ui("Generation failed.", "生成失败。");
  } finally {
    generateButton.querySelector("span:first-child").textContent = ui("Generate polymers", "生成聚合物");
    generateButton.disabled = !models.has(modelKey);
  }
}

generateButton.addEventListener("click", generate);
document.querySelectorAll('.header-nav [aria-disabled="true"]').forEach((link) => {
  link.addEventListener("click", (event) => event.preventDefault());
});
loadModels();
