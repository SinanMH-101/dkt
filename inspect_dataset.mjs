import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const input = await FileBlob.load("Sinan_Dataset.xlsx");
const workbook = await SpreadsheetFile.importXlsx(input);
const summary = await workbook.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
console.log(summary.ndjson);
