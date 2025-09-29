import os
import glob
import pandas as pd
import bibtexparser
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn

# ==============================
# CONFIGURACIÓN
# ==============================
INPUT_DIRS = [
    r"C:\Users\clare\Downloads\IEEE",
    r"C:\Users\clare\Downloads\ScienceDirect"
]
OUTPUT_DIR = r"C:\Users\clare\bibliometria\DataFinal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

console = Console()

# ==============================
# Columnas finales (sin Publisher y Language)
# ==============================
COLUMNAS = [
    "Authors", "Title", "Year", "Volume", "Issue",
    "Start Page", "End Page", "Abstract", "DOI",
    "Author Keywords", "Publication Title", "Link", "Database"
]

# ==============================
# Función para leer un .bib con "Desconocido"
# ==============================
def leer_bibtex(archivo, fuente):
    with open(archivo, encoding="utf-8", errors="ignore") as f:
        bib_database = bibtexparser.load(f)

    datos = []
    for e in bib_database.entries:
        # Páginas separadas si existen
        pages = e.get("pages", "").split("-")
        start_page = pages[0] if len(pages) > 0 and pages[0] else "Desconocido"
        end_page = pages[1] if len(pages) > 1 and pages[1] else "Desconocido"

        datos.append({
            "Authors": e.get("author", "Desconocido").replace("\n", " ").replace(",", ";"),
            "Title": e.get("title", "Desconocido"),
            "Year": e.get("year", "Desconocido"),
            "Volume": e.get("volume", "Desconocido"),
            "Issue": e.get("number", "Desconocido"),
            "Start Page": start_page,
            "End Page": end_page,
            "Abstract": e.get("abstract", "Desconocido"),
            "DOI": e.get("doi", "Desconocido"),
            "Author Keywords": e.get("keywords", "Desconocido").replace(",", ";"),
            "Publication Title": e.get("journal", e.get("booktitle", "Desconocido")),
            "Link": e.get("url") if e.get("url") else f"https://doi.org/{e.get('doi', '')}" if e.get("doi") else "Desconocido",
            "Database": fuente
        })

    return datos

# ==============================
# 1. Leer todos los archivos .bib
# ==============================
console.print("\n[bold cyan]*** Sistema Unificador de Datos Bibliográficos ***[/bold cyan]\n")

todos_datos = []
with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(bar_width=40),
    "[progress.percentage]{task.percentage:>3.0f}%", TimeElapsedColumn(),
    console=console
) as progress:
    total_archivos = sum(len(glob.glob(os.path.join(c, "*.bib"))) for c in INPUT_DIRS)
    task = progress.add_task("Unificando archivos .bib y creando .csv", total=total_archivos)

    for carpeta in INPUT_DIRS:
        archivos_bib = glob.glob(os.path.join(carpeta, "*.bib"))
        for archivo in archivos_bib:
            fuente = os.path.basename(carpeta)
            todos_datos.extend(leer_bibtex(archivo, fuente))
            progress.advance(task)

# Convertir a DataFrame y reemplazar vacíos con "Desconocido"
df = pd.DataFrame(todos_datos, columns=COLUMNAS)
df = df.fillna("Desconocido")  # <-- Aquí se aseguran todos los vacíos

# ==============================
# 2. Detectar duplicados
# ==============================
df["Title_lower"] = df["Title"].str.lower().str.strip()
duplicados = df[df.duplicated(subset=["DOI", "Title_lower"], keep=False)]
duplicados.to_csv(os.path.join(OUTPUT_DIR, "duplicados.csv"), sep=";", index=False, encoding="utf-8-sig")

df_unico = df.drop_duplicates(subset=["DOI", "Title_lower"], keep="first")
df_unico = df_unico.drop(columns=["Title_lower"])
df_unico.to_csv(os.path.join(OUTPUT_DIR, "unificado.csv"), sep=";", index=False, encoding="utf-8-sig")

# ==============================
# 3. Mostrar estadísticas
# ==============================
console.print("\n[bold magenta]========================================[/bold magenta]")
console.print(f"[bold green]Archivos CSV guardados en: {OUTPUT_DIR}[/bold green]\n")

conteo_fuentes = Counter(df["Database"])
tabla = Table(title="Conteo de artículos por fuente", show_lines=True)
tabla.add_column("Fuente", style="cyan", justify="center")
tabla.add_column("Total Artículos", style="green", justify="center")

for fuente, total in conteo_fuentes.items():
    tabla.add_row(fuente, str(total))

tabla.add_row("TOTAL", str(len(df)))
tabla.add_row("DUPLICADOS", str(len(duplicados)))
tabla.add_row("FINAL (SIN DUPLICADOS)", str(len(df_unico)))

console.print(tabla)

console.print(f"[bold red]Número de entradas duplicadas antes de la eliminación: {len(duplicados)}[/bold red]")
console.print("[bold magenta]========================================[/bold magenta]")
