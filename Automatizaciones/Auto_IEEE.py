import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.common.exceptions import TimeoutException

# ===================== CONFIG =====================
CHROMEDRIVER = r"C:\Users\clare\bibliometria\chromedriver\chromedriver.exe"
TEMP_PROFILE = r"C:\Users\clare\bibliometria\chrome_temp_profile"
DOWNLOAD_DIR = r"C:\Users\clare\Downloads"
LIBRARY_URL = "https://library.uniquindio.edu.co/databases"
SEARCH_QUERY = '"generative artificial intelligence"'
NUM_PAGES = 20  

# ===================== SELENIUM SETUP =====================
options = webdriver.ChromeOptions()
options.add_argument(f"user-data-dir={TEMP_PROFILE}")
options.add_experimental_option("detach", True)
options.add_argument("--disable-popup-blocking")
options.add_argument("--disable-notifications")
options.add_argument("--no-first-run")
options.add_argument("--no-default-browser-check")
prefs = {"download.default_directory": DOWNLOAD_DIR, "download.prompt_for_download": False}
options.add_experimental_option("prefs", prefs)

service = Service(CHROMEDRIVER)
driver = webdriver.Chrome(service=service, options=options)
driver.maximize_window()
wait = WebDriverWait(driver, 20)

# ===================== FUNCIONES AUXILIARES =====================
def close_download_modal(wait_after=7):
    """Cerrar modal de descarga con ESCAPE después de esperar algunos segundos"""
    time.sleep(wait_after)
    ActionChains(driver).send_keys(Keys.ESCAPE).perform()
    print(f"Modal cerrado después de {wait_after} segundos")

def wait_for_download(file_extension=".bib", timeout=30):
    """Esperar que aparezca un archivo descargado en la carpeta de descargas"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        for filename in os.listdir(DOWNLOAD_DIR):
            if filename.endswith(file_extension):
                print(f" Archivo descargado: {filename}")
                return True
        time.sleep(1)
    print("Tiempo de espera excedido. Archivo no descargado.")
    return False

def download_bibtex(retries=3):
    """Selecciona todos, abre Export y descarga BibTeX, reintentando si no se descarga"""
    try:
        select_all = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//input[contains(@class,'results-actions-selectall-checkbox')]")))
        if not select_all.is_selected():
            driver.execute_script("arguments[0].click();", select_all)

        export_btn = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(),'Export')]")))
        time.sleep(1)
        driver.execute_script("arguments[0].click();", export_btn)

        citations_tab = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//a[contains(text(),'Citations')]")))
        time.sleep(1)
        driver.execute_script("arguments[0].click();", citations_tab)

        download_radios = wait.until(EC.presence_of_all_elements_located((By.NAME, "download-format")))
        driver.execute_script("arguments[0].click();", download_radios[1])
        citations_radios = wait.until(EC.presence_of_all_elements_located((By.NAME, "citations-format")))
        driver.execute_script("arguments[0].click();", citations_radios[1])

        for attempt in range(1, retries + 1):
            print(f"Intento de descarga {attempt}...")
            download_btn = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(@class,'stats-SearchResults_Citation_Download')]")))
            time.sleep(3)
            driver.execute_script("arguments[0].click();", download_btn)
            close_download_modal(wait_after=6)
            if wait_for_download(".bib", timeout=10):
                print("Descarga exitosa.")
                break
            else:
                print("⚠ Archivo no descargado, reintentando...")
                time.sleep(2)
        else:
            print("No se pudo descargar el archivo después de varios intentos.")
    except Exception as e:
        print(f"Error en download_bibtex: {e}")

# ===================== FLUJO PRINCIPAL =====================
driver.get(LIBRARY_URL)
time.sleep(2)

# Abrir Fac. Ingeniería
fac_ing_btn = wait.until(EC.presence_of_element_located(
    (By.XPATH, "//h2[contains(text(),'Fac. Ingeniería')]")))
driver.execute_script("arguments[0].click();", fac_ing_btn)
time.sleep(2)

# Abrir IEEE
ieee_xpath = "//span[contains(translate(., 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'IEEE')]"
driver.execute_script("arguments[0].click();", driver.find_element(By.XPATH, ieee_xpath))
time.sleep(3)

# Login Google si no hay sesión
google_btn = wait.until(EC.element_to_be_clickable((By.ID, "btn-google")))
google_btn.click()
time.sleep(3)
try:
    search_box = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//input[@type='search' and contains(@class,'Typeahead-input')]"))
    )
except TimeoutException:
    email_input = wait.until(EC.presence_of_element_located((By.ID, "identifierId")))
    email_input.send_keys("Correo", Keys.ENTER)
    time.sleep(3)
    password_input = wait.until(EC.presence_of_element_located((By.NAME, "Passwd")))
    password_input.send_keys("Contraseña", Keys.ENTER)
    time.sleep(5)

# Búsqueda
search_box = wait.until(EC.presence_of_element_located(
    (By.XPATH, "//input[@type='search' and contains(@class,'Typeahead-input')]")))
search_box.clear()
search_box.send_keys(SEARCH_QUERY, Keys.ENTER)
time.sleep(5)

# ===================== RECORRER PÁGINAS =====================
for page in range(1, NUM_PAGES + 1):
    print(f"\n=== Procesando página {page} ===")
    try:
        if page == 1:
            # Página 1: cambiar a 100 resultados
            items_btn = wait.until(EC.element_to_be_clickable((By.ID, "dropdownPerPageLabel")))
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", items_btn)
            items_btn.click()
            time.sleep(1)
            option_100 = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(text(),'100')]")))
            option_100.click()
            time.sleep(3)

            # Descargar en la página 1
            download_bibtex()

        elif 2 <= page <= 10:
            # Páginas 2 a 10: botón numérico directo
            page_btn_xpath = f"//button[contains(@class,'stats-Pagination_{page}')]"
            page_btn = wait.until(EC.element_to_be_clickable((By.XPATH, page_btn_xpath)))
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", page_btn)
            driver.execute_script("arguments[0].click();", page_btn)
            time.sleep(5)

            download_bibtex()

        elif page == 11:
            # Página 11: usar botón Next para mostrar la numeración 11-15 y navegar a 11
            next_btn = wait.until(EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(@class,'stats-Pagination_arrow_next_11')]")))
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_btn)
            driver.execute_script("arguments[0].click();", next_btn)
            time.sleep(5)

            # ahora estamos en la página 11; descargar
            download_bibtex()

        else:
            # Páginas 12..NUM_PAGES: boton numerico ahora visible
            page_btn_xpath = f"//button[contains(@class,'stats-Pagination_{page}')]"
            page_btn = wait.until(EC.element_to_be_clickable((By.XPATH, page_btn_xpath)))
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", page_btn)
            driver.execute_script("arguments[0].click();", page_btn)
            time.sleep(5)

            download_bibtex()

    except Exception as e:
        print(f"⚠ No se pudo procesar la página {page}: {e}")

print("\n Proceso completo en IEEE Xplore terminado.")
