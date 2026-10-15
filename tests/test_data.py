import json
import zipfile
from xml.sax.saxutils import escape
import pandas as pd
from il_contrastive import data
from il_contrastive.tasks import TASKS


def write_xlsx(path, rows):
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    contents = []
    for i, row in enumerate(rows, 1):
        cells = ''.join(f'<c r="{chr(65+j)}{i}" t="inlineStr"><is><t>{escape(str(v))}</t></is></c>'
                        for j, v in enumerate(row))
        contents.append(f'<row r="{i}">{cells}</row>')
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('xl/workbook.xml', f'<workbook xmlns="{ns}" xmlns:r="{rel}"><sheets><sheet name="lookup" r:id="rId1"/><sheet name="co2solubility" r:id="rId2"/></sheets></workbook>')
        z.writestr('xl/_rels/workbook.xml.rels', '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/><Relationship Id="rId2" Target="worksheets/sheet2.xml"/></Relationships>')
        z.writestr('xl/worksheets/sheet1.xml', f'<worksheet xmlns="{ns}"><sheetData/></worksheet>')
        z.writestr('xl/worksheets/sheet2.xml', f'<worksheet xmlns="{ns}"><sheetData>{"".join(contents)}</sheetData></worksheet>')


def test_sheet_selection_conversion_quarantine_and_dedup(tmp_path, monkeypatch):
    spec = TASKS['co2_solubility']
    monkeypatch.setattr(data, 'TASKS', {'co2_solubility':spec})
    s = 'CCCC[n+]1ccn(C)c1.[Cl-]'
    row = [s, .3, 300, 10]
    rows = [['IL SMILES', 'x_CO2', 'T (K)', 'P (bar)'], row, row,
            ['CC.[Cl-]', .2, 300, 10], [s, .4, 320, 20]]
    write_xlsx(tmp_path / spec['file'], rows)
    output = tmp_path / 'processed'
    report = data.prepare(tmp_path, output)
    result = pd.read_csv(output / 'observations.csv.gz')
    assert len(result) == 2
    assert result.pressure_input.tolist() == [1., 2.]
    assert result.split.nunique() == 1
    assert report['rejected'] == 1
    assert report['exact_duplicates_removed'] == 1
    assert result.source_row.tolist() == [2, 5]
    assert result.source_sheet.unique().tolist() == ['co2solubility']
