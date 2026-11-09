import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

// Run from the repository root to list the local COMP1350 workbook's worksheets.
const input = await FileBlob.load("Sinan_Dataset.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
const summary = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
console.log(summary.ndjson);
