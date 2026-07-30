import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const analysisRoot = path.resolve(process.argv[2]);
const outputPath = path.resolve(process.argv[3]);
const previewDir = path.resolve(process.argv[4]);

const COLORS = {
  navy: "#17324D",
  teal: "#007C83",
  lightTeal: "#DCEFF0",
  pale: "#F4F7F9",
  border: "#D6DEE5",
  dark: "#1F2933",
  gray: "#667785",
  white: "#FFFFFF",
  y2016: "#0072B2",
  y2020: "#E69F00",
  y2024: "#009E73",
};

const dataSheets = [
  {
    name: "Pelotas",
    file: "tabelas/resumo_pelotas_dinheiro_votos.csv",
    table: "TblPelotas",
  },
  {
    name: "Dinheiro_Votos",
    file: "tabelas/modelos_dinheiro_votos.csv",
    table: "TblDinheiroVotos",
  },
  {
    name: "Eleicao_LPM",
    file: "tabelas/modelos_eleicao_lpm.csv",
    table: "TblEleicaoLPM",
  },
  {
    name: "Atenuacao",
    file: "tabelas/atenuacao_coeficiente_dinheiro.csv",
    table: "TblAtenuacao",
  },
  {
    name: "Validacao_CV",
    file: "tabelas/validacao_cruzada_pelotas_resumo.csv",
    table: "TblValidacaoCV",
  },
  {
    name: "Contrastes_CV",
    file: "tabelas/validacao_cruzada_pelotas_contrastes.csv",
    table: "TblContrastesCV",
  },
  {
    name: "Logit_Condicional",
    file: "tabelas/modelos_eleicao_coeficientes.csv",
    table: "TblLogitCondicional",
  },
  {
    name: "Painel_Pelotas",
    file: "tabelas/painel_candidatos_pelotas_votos.csv",
    table: "TblPainelPelotas",
  },
  {
    name: "Municipios",
    file: "tabelas/indicadores_municipio_dinheiro_votos.csv",
    table: "TblMunicipios",
  },
  {
    name: "Forca_Listas",
    file: "tabelas/forca_listas_rs.csv",
    table: "TblForcaListas",
  },
  {
    name: "Auditoria",
    file: "tabelas/auditoria_votos.csv",
    table: "TblAuditoria",
  },
];

function excelColumn(index) {
  let value = index + 1;
  let result = "";
  while (value > 0) {
    value -= 1;
    result = String.fromCharCode(65 + (value % 26)) + result;
    value = Math.floor(value / 26);
  }
  return result;
}

async function csvValues(relativePath) {
  const csvText = await fs.readFile(
    path.join(analysisRoot, relativePath),
    "utf8",
  );
  const imported = await Workbook.fromCSV(csvText, { sheetName: "Import" });
  const sheet = imported.worksheets.getItem("Import");
  return sheet.getUsedRange(true).values;
}

function chooseNumberFormat(header) {
  const normalized = String(header).toLowerCase();
  if (normalized === "ano") {
    return "0";
  }
  if (
    normalized.includes("p_valor") ||
    normalized.startsWith("p_") ||
    normalized.includes("gradiente")
  ) {
    return "0.000E+00";
  }
  if (
    normalized.includes("receita_total") ||
    normalized.includes("despesa_total") ||
    normalized.includes("_2024") ||
    normalized.includes("_nominal")
  ) {
    return 'R$ #,##0.00';
  }
  if (
    normalized.includes("atenuacao") ||
    normalized.includes("fracao_") ||
    normalized.includes("percentil_")
  ) {
    return "0.0%";
  }
  if (
    normalized.includes("rho_") ||
    normalized.includes("auc") ||
    normalized.includes("beta") ||
    normalized.includes("r2_") ||
    normalized.includes("multiplicador") ||
    normalized.includes("odds_ratio") ||
    normalized.includes("delta_")
  ) {
    return "0.000";
  }
  if (
    normalized.includes("candidatos") ||
    normalized.includes("eleitos") ||
    normalized.includes("votos") ||
    normalized.includes("linhas") ||
    normalized === "n" ||
    normalized.includes("municipios") ||
    normalized.includes("grupos") ||
    normalized.includes("folds") ||
    normalized.includes("zonas")
  ) {
    return "#,##0";
  }
  return null;
}

function styleDataSheet(sheet, values, tableName) {
  const rowCount = values.length;
  const colCount = values[0].length;
  const lastColumn = excelColumn(colCount - 1);
  const used = sheet.getRange(`A1:${lastColumn}${rowCount}`);
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);
  used.format.font = { name: "Aptos", size: 10, color: COLORS.dark };
  used.format.borders = {
    insideHorizontal: { style: "thin", color: "#E8EDF1" },
  };
  sheet.getRange(`A1:${lastColumn}1`).format = {
    fill: COLORS.navy,
    font: { name: "Aptos", size: 10, bold: true, color: COLORS.white },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "all", style: "thin", color: COLORS.navy },
    rowHeight: 42,
  };
  used.format.columnWidth = 14;
  const headers = values[0].map((value) => String(value));
  headers.forEach((header, columnIndex) => {
    const column = excelColumn(columnIndex);
    const textLike = /(nome|modelo|parametro|comparacao|contraste|metodo|arquivo|municipio|partido|lista_id|situacao|ocupacao|composicao)/i.test(
      header,
    );
    let width = textLike ? 26 : 14;
    if (/(arquivo|composicao)/i.test(header)) width = 38;
    if (!textLike && header.length > 20) width = 18;
    sheet.getRange(`${column}:${column}`).format.columnWidth = width;
    const numberFormat = chooseNumberFormat(header);
    if (numberFormat && rowCount > 1) {
      sheet
        .getRange(`${column}2:${column}${rowCount}`)
        .setNumberFormat(numberFormat);
    }
  });
  const table = sheet.tables.add(
    `A1:${lastColumn}${rowCount}`,
    true,
    tableName,
  );
  table.style = "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;
}

function addNarrativeRow(sheet, range, text, fill) {
  const target = sheet.getRange(range);
  target.merge();
  target.values = [[text]];
  target.format = {
    fill,
    font: { name: "Aptos", size: 10, color: COLORS.dark },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: COLORS.border },
  };
}

const workbook = Workbook.create();
const summary = workbook.worksheets.add("Resumo");

for (const spec of dataSheets) {
  const sheet = workbook.worksheets.add(spec.name);
  const values = await csvValues(spec.file);
  const target = sheet.getRangeByIndexes(
    0,
    0,
    values.length,
    values[0].length,
  );
  target.values = values;
  styleDataSheet(sheet, values, spec.table);
}

const methodology = workbook.worksheets.add("Metodologia");
const figures = workbook.worksheets.add("Figuras");

summary.showGridLines = false;
summary.getRange("A1:N2").merge();
summary.getRange("A1").values = [[
  "Teste do acoplamento entre dinheiro, votos e eleição",
]];
summary.getRange("A1:N2").format = {
  fill: COLORS.navy,
  font: { name: "Aptos Display", size: 20, bold: true, color: COLORS.white },
  horizontalAlignment: "left",
  verticalAlignment: "center",
};
summary.getRange("A3:N3").merge();
summary.getRange("A3").values = [[
  "Vereadores • Pelotas e 497 municípios do RS • eleições de 2016, 2020 e 2024",
]];
summary.getRange("A3:N3").format = {
  fill: COLORS.lightTeal,
  font: { name: "Aptos", size: 11, italic: true, color: COLORS.navy },
  verticalAlignment: "center",
};

summary.getRange("A5:D12").values = [
  ["Métrica", "2016", "2020", "2024"],
  ["ρ receita–votos (Pelotas)", null, null, null],
  ["ρ dentro da lista (Pelotas)", null, null, null],
  ["AUC receita → eleição", null, null, null],
  ["AUC votos → eleição", null, null, null],
  ["Atenuação após votos + lista (RS)", null, null, null],
  ["ΔAUC dinheiro após viabilidade prévia", null, null, null],
  ["ΔAUC dinheiro após votos + lista", null, null, null],
];
summary.getRange("B6:D6").formulas = [[
  "=Pelotas!E2",
  "=Pelotas!E3",
  "=Pelotas!E4",
]];
summary.getRange("B7:D7").formulas = [[
  "=Pelotas!I2",
  "=Pelotas!I3",
  "=Pelotas!I4",
]];
summary.getRange("B8:D8").formulas = [[
  "=Pelotas!K2",
  "=Pelotas!K3",
  "=Pelotas!K4",
]];
summary.getRange("B9:D9").formulas = [[
  "=Pelotas!M2",
  "=Pelotas!M3",
  "=Pelotas!M4",
]];
summary.getRange("B10:D10").formulas = [[
  "=Atenuacao!G2",
  "=Atenuacao!G3",
  "=Atenuacao!G4",
]];
summary.getRange("B11:D11").values = [[null, null, null]];
summary.getRange("C11:D11").formulas = [[
  "=Contrastes_CV!E4",
  "=Contrastes_CV!E6",
]];
summary.getRange("B12:D12").formulas = [[
  "=Contrastes_CV!E2",
  "=Contrastes_CV!E3",
  "=Contrastes_CV!E5",
]];
summary.getRange("A5:D5").format = {
  fill: COLORS.teal,
  font: { name: "Aptos", size: 11, bold: true, color: COLORS.white },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange("A6:A12").format = {
  fill: COLORS.pale,
  font: { name: "Aptos", size: 10, bold: true, color: COLORS.dark },
  wrapText: true,
};
summary.getRange("A5:D12").format.borders = {
  preset: "all",
  style: "thin",
  color: COLORS.border,
};
summary.getRange("B6:D9").setNumberFormat("0.000");
summary.getRange("B10:D10").setNumberFormat("0.0%");
summary.getRange("B11:D12").setNumberFormat("0.000");
summary.getRange("A:A").format.columnWidth = 37;
summary.getRange("B:D").format.columnWidth = 14;
summary.getRange("F:N").format.columnWidth = 12;

summary.getRange("A15:D15").merge();
summary.getRange("A15").values = [["Leitura do teste"]];
summary.getRange("A15:D15").format = {
  fill: COLORS.navy,
  font: { name: "Aptos", size: 12, bold: true, color: COLORS.white },
  verticalAlignment: "center",
};
addNarrativeRow(
  summary,
  "A16:D17",
  "1. Recursos e votos: a receita se associa fortemente aos votos em Pelotas e a relação permanece dentro das listas.",
  "#F7FAFC",
);
addNarrativeRow(
  summary,
  "A18:D19",
  "2. Mecanismo: no RS, votos nominais e força da lista absorvem 74,8%–93,2% do coeficiente inicial do dinheiro.",
  "#EEF7F5",
);
addNarrativeRow(
  summary,
  "A20:D21",
  "3. Seleção reversa: o dinheiro melhora a previsão baseada em viabilidade anterior, mas não depois que os votos correntes já são conhecidos.",
  "#FFF7E6",
);
addNarrativeRow(
  summary,
  "A22:D24",
  "Cautela: a análise é observacional. Votos e força corrente da lista são pós-campanha; a atenuação é evidência compatível com mecanismo, não uma estimativa causal.",
  "#FDECEC",
);

summary.getRange("A27:D28").values = [
  ["Métrica", "2016", "2020", "2024"],
  ["Atenuação do coeficiente", null, null, null],
];
summary.getRange("B28:D28").formulas = [[
  "=Atenuacao!G2",
  "=Atenuacao!G3",
  "=Atenuacao!G4",
]];
summary.getRange("A27:D27").format = {
  fill: COLORS.teal,
  font: { name: "Aptos", size: 10, bold: true, color: COLORS.white },
};
summary.getRange("A27:D28").format.borders = {
  preset: "all",
  style: "thin",
  color: COLORS.border,
};
summary.getRange("B28:D28").setNumberFormat("0.0%");

const aucChart = summary.charts.add("bar", {
  chartType: "bar",
  title: "Pelotas: associação e discriminação eleitoral",
  hasLegend: true,
});
aucChart.title = "Pelotas: associação e discriminação eleitoral";
aucChart.hasLegend = true;
aucChart.setPosition("F5", "N20");
aucChart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 9 } };
aucChart.yAxis = { numberFormatCode: "0.0", min: 0, max: 1 };
const yearFills = [COLORS.y2016, COLORS.y2020, COLORS.y2024];
["2016", "2020", "2024"].forEach((year, index) => {
  const series = aucChart.series.add(year);
  series.categoryFormula = "'Resumo'!$A$6:$A$9";
  series.formula = `'Resumo'!$${excelColumn(index + 1)}$6:$${excelColumn(index + 1)}$9`;
  series.fill = yearFills[index];
});

const attenuationChart = summary.charts.add("bar", {
  chartType: "bar",
  title: "RS: atenuação após votos e força da lista",
  hasLegend: true,
});
attenuationChart.title = "RS: atenuação após votos e força da lista";
attenuationChart.hasLegend = true;
attenuationChart.setPosition("F22", "N36");
attenuationChart.yAxis = { numberFormatCode: "0%", min: 0, max: 1 };
["2016", "2020", "2024"].forEach((year, index) => {
  const series = attenuationChart.series.add(year);
  series.categoryFormula = "'Resumo'!$A$28:$A$28";
  series.formula = `'Resumo'!$${excelColumn(index + 1)}$28:$${excelColumn(index + 1)}$28`;
  series.fill = yearFills[index];
});
summary.freezePanes.freezeRows(3);

const methodologyJson = JSON.parse(
  await fs.readFile(path.join(analysisRoot, "metodologia.json"), "utf8"),
);
const methodologyRows = [
  ["Definição", "Descrição"],
  ["Universo", methodologyJson.universo],
  ["Votos nominais", methodologyJson.votos_nominais],
  ["Força da lista — 2016", methodologyJson.forca_lista["2016"]],
  ["Força da lista — 2020", methodologyJson.forca_lista["2020"]],
  ["Força da lista — 2024", methodologyJson.forca_lista["2024"]],
  ["Fórmula da força da lista", methodologyJson.forca_lista.formula],
  ["Viabilidade prévia", methodologyJson.viabilidade_previa],
  ["Modelo dinheiro → votos", methodologyJson.modelo_dinheiro_votos],
  ["Modelos de eleição", methodologyJson.modelo_eleicao],
  ["Separação quase perfeita", methodologyJson.separacao],
  ["Cautela causal", methodologyJson.cautela],
  ["Fonte TSE — 2016", methodologyJson.fontes["2016"]],
  ["Fonte TSE — 2020", methodologyJson.fontes["2020"]],
  ["Fonte TSE — 2024", methodologyJson.fontes["2024"]],
];
methodology.getRangeByIndexes(
  0,
  0,
  methodologyRows.length,
  2,
).values = methodologyRows;
methodology.showGridLines = false;
methodology.freezePanes.freezeRows(1);
methodology.getRange(`A1:B${methodologyRows.length}`).format = {
  font: { name: "Aptos", size: 10, color: COLORS.dark },
  wrapText: true,
  verticalAlignment: "top",
  borders: { preset: "all", style: "thin", color: COLORS.border },
};
methodology.getRange("A1:B1").format = {
  fill: COLORS.navy,
  font: { name: "Aptos", size: 11, bold: true, color: COLORS.white },
};
methodology.getRange(`A2:A${methodologyRows.length}`).format = {
  fill: COLORS.lightTeal,
  font: { name: "Aptos", size: 10, bold: true, color: COLORS.navy },
};
methodology.getRange("A:A").format.columnWidth = 28;
methodology.getRange("B:B").format.columnWidth = 95;
methodology.getRange(`A2:B${methodologyRows.length}`).format.rowHeight = 42;

figures.showGridLines = false;
figures.getRange("A1:N2").merge();
figures.getRange("A1").values = [["Figuras do teste"]];
figures.getRange("A1:N2").format = {
  fill: COLORS.navy,
  font: { name: "Aptos Display", size: 18, bold: true, color: COLORS.white },
  verticalAlignment: "center",
};
const imageSpecs = [
  ["figura_1_dinheiro_votos_pelotas.png", 3, 0, 1150, 385],
  ["figura_2_auc_dinheiro_votos_lista.png", 27, 0, 900, 470],
  ["figura_3_atenuacao_dinheiro.png", 57, 0, 1150, 390],
  ["figura_4_validacao_viabilidade.png", 82, 0, 1000, 470],
];
for (const [filename, row, col, widthPx, heightPx] of imageSpecs) {
  const bytes = await fs.readFile(path.join(analysisRoot, "figuras", filename));
  figures.images.add({
    dataUrl: `data:image/png;base64,${bytes.toString("base64")}`,
    anchor: {
      from: { row, col },
      extent: { widthPx, heightPx },
    },
  });
}

await fs.mkdir(previewDir, { recursive: true });
const renderSpecs = [
  ["Resumo", "A1:N36"],
  ["Pelotas", "A1:R6"],
  ["Dinheiro_Votos", "A1:N14"],
  ["Eleicao_LPM", "A1:L25"],
  ["Atenuacao", "A1:G6"],
  ["Validacao_CV", "A1:G18"],
  ["Contrastes_CV", "A1:J8"],
  ["Logit_Condicional", "A1:O25"],
  ["Painel_Pelotas", "A1:Z22"],
  ["Municipios", "A1:O25"],
  ["Forca_Listas", "A1:N25"],
  ["Auditoria", "A1:Q6"],
  ["Metodologia", "A1:B15"],
  ["Figuras", "A1:N112"],
];

const formulaErrors = [];
for (const sheetName of workbook.worksheets.items.map((sheet) => sheet.name)) {
  const sheet = workbook.worksheets.getItem(sheetName);
  const used = sheet.getUsedRange(true);
  if (!used) continue;
  const values = used.values;
  values.forEach((row, rowIndex) => {
    row.forEach((value, colIndex) => {
      if (
        typeof value === "string" &&
        /^#(REF!|DIV\/0!|VALUE!|NAME\?|N\/A|NUM!|NULL!)/.test(value)
      ) {
        formulaErrors.push({
          sheet: sheetName,
          cell: `${excelColumn(colIndex)}${rowIndex + 1}`,
          value,
        });
      }
    });
  });
}
if (formulaErrors.length) {
  throw new Error(`Erros de fórmula: ${JSON.stringify(formulaErrors)}`);
}

for (const [sheetName, range] of renderSpecs) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale: sheetName === "Resumo" ? 1.2 : 0.8,
    format: "png",
  });
  await fs.writeFile(
    path.join(previewDir, `${sheetName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

const inspection = await workbook.inspect({
  kind: "formula",
  sheetId: "Resumo",
  range: "A1:N36",
  maxChars: 12000,
  options: { maxResults: 100 },
});
await fs.writeFile(
  path.join(previewDir, "inspecao_formulas_resumo.ndjson"),
  inspection.ndjson,
  "utf8",
);

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);
console.log(JSON.stringify({
  outputPath,
  sheets: workbook.worksheets.items.map((sheet) => sheet.name),
  formulaErrors: formulaErrors.length,
  previews: renderSpecs.length,
}, null, 2));
