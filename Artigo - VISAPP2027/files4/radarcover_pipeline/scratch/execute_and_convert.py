import os
import sys
import subprocess
import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

notebook_path = "/Users/mm/mestrado/Artigo - VISAPP2027/files4/radarcover_pipeline/exploracao_dados_radarcover.ipynb"
executed_notebook_path = "/Users/mm/mestrado/Artigo - VISAPP2027/files4/radarcover_pipeline/exploracao_dados_radarcover_executed.ipynb"
html_path = "/Users/mm/mestrado/Artigo - VISAPP2027/files4/radarcover_pipeline/exploracao_dados_radarcover.html"
pdf_path = "/Users/mm/mestrado/Artigo - VISAPP2027/files4/radarcover_pipeline/exploracao_dados_radarcover.pdf"

print("1. Carregando notebook...")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

print("2. Executando todas as células do notebook...")
client = NotebookClient(nb, timeout=1200)
executed_nb = client.execute()

print(f"3. Salvando notebook executado em: {executed_notebook_path}")
with open(executed_notebook_path, "w", encoding="utf-8") as f:
    nbformat.write(executed_nb, f)

print("4. Convertendo notebook executado para HTML...")
html_exporter = HTMLExporter()
html_exporter.template_name = "classic"
(body, resources) = html_exporter.from_notebook_node(executed_nb)

# Adiciona estilos CSS customizados para melhorar o layout da impressão/PDF
custom_css = """
<style>
@media print {
    @page {
        size: A4 portrait;
        margin: 1.5cm;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        font-size: 10pt;
    }
    .jp-Cell, .cell {
        page-break-inside: avoid;
        margin-bottom: 1.5em;
    }
    img {
        max-width: 100% !important;
        height: auto !important;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 1em;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 6px 10px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
}
</style>
"""
body = body.replace("</head>", f"{custom_css}\n</head>")

with open(html_path, "w", encoding="utf-8") as f:
    f.write(body)

print(f"5. Gerando PDF com Chrome Headless em: {pdf_path}")
chrome_cmd = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "--headless",
    "--disable-gpu",
    "--no-pdf-header-footer",
    f"--print-to-pdf={pdf_path}",
    html_path
]

res = subprocess.run(chrome_cmd, capture_output=True, text=True)
if res.returncode == 0:
    print(f"✓ PDF gerado com sucesso em: {pdf_path}")
else:
    print(f"Erro ao gerar PDF: {res.stderr}")
