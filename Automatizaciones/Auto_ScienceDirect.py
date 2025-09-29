import os
import shutil
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, TimeoutException

# ==============================
# CONFIGURACIÓN
# ==============================
ORIGINAL_PROFILE = r"C:\Users\clare\AppData\Local\Google\Chrome\User Data\Profile 2"
TEMP_PROFILE = r"C:\Users\clare\bibliometria\chrome_temp_profile"
CHROMEDRIVER = r"C:\Users\clare\bibliometria\chromedriver\chromedriver.exe"
LIBRARY_URL = "https://library.uniquindio.edu.co/databases"
MAX_PAGES = 50  

# ==============================
# FUNCIONES RANDOMS
# ==============================
def pausa_random(min_s=4, max_s=9):
    """Pausa con tiempo aleatorio entre min_s y max_s segundos"""
    t = random.uniform(min_s, max_s)
    print(f" Esperando {t:.1f} segundos.")
    time.sleep(t)

# ==============================
# UTILIDADES
# ==============================
def limpiar_perfil():
    if os.path.exists(TEMP_PROFILE):
        print(" Eliminando perfil temporal viejo.")
        shutil.rmtree(TEMP_PROFILE)

def copiar_perfil():
    print(" Copiando Profile 2 a perfil temporal.")
    def ignore_in_use(dir, files):
        return [f for f in files if "Cookies" in f or "Sessions" in f or "Safe Browsing" in f]
    shutil.copytree(ORIGINAL_PROFILE, TEMP_PROFILE, ignore=ignore_in_use)

def configurar_driver():
    options = webdriver.ChromeOptions()
    options.add_argument(f"user-data-dir={TEMP_PROFILE}")
    options.add_experimental_option("detach", True)
    options.add_argument("--disable-features=ChromeLabs,EnableEphemeralFlashPermission")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-notifications")
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.add_argument("--disable-features=ChromeWhatsNewUI,SignInProfileCreation")

    # Quitar "Iniciar sin chrome"
    prefs = {
        "signin.allowed_on_next_startup": False,
        "credentials_enable_service": False,
        "profile.password_manager_enabled": False
    }
    options.add_experimental_option("prefs", prefs)

    options.add_argument("--disable-features=BrowserSignin")


    service = Service(CHROMEDRIVER)
    driver = webdriver.Chrome(service=service, options=options)
    driver.maximize_window()
    return driver

# ==============================
# FLUJO PRINCIPAL
# ==============================
def abrir_biblioteca(driver, wait):
    driver.get(LIBRARY_URL)
    print(" Biblioteca abierta")
    pausa_random()

    print(" Abriendo 'Fac. Ingeniería'...")
    fac_ing_btn = wait.until(
        EC.presence_of_element_located((By.XPATH, "//h2[contains(text(),'Fac. Ingeniería')]"))
    )
    driver.execute_script("arguments[0].scrollIntoView(true);", fac_ing_btn)
    pausa_random()
    driver.execute_script("arguments[0].click();", fac_ing_btn)

def abrir_sciencedirect(driver):
    print(" Buscando 'SCIENCEDIRECT - (DESCUBRIDOR)'...")
    found = False
    scroll_attempts = 0
    while not found and scroll_attempts < 20:
        try:
            scidir = driver.find_element(
                By.XPATH,
                "//span[contains(translate(., 'abcdefghijklmnopqrstuvwxyz', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 'SCIENCEDIRECT')]"
            )
            driver.execute_script("arguments[0].scrollIntoView({behavior: 'smooth', block: 'center'});", scidir)
            pausa_random()
            driver.execute_script("arguments[0].click();", scidir)
            print(" ScienceDirect abierto.")
            found = True
        except (ElementClickInterceptedException, TimeoutException):
            driver.execute_script("window.scrollBy(0, 300);")
            time.sleep(0.5)
            scroll_attempts += 1
    if not found:
        print("⚠ No se pudo abrir ScienceDirect.")

def login_google(driver, wait):
    print(" Iniciando sesión con Google...")
    google_btn = wait.until(EC.element_to_be_clickable((By.ID, "btn-google")))
    google_btn.click()

    email_input = wait.until(EC.presence_of_element_located((By.ID, "identifierId")))
    email_input.clear()
    email_input.send_keys("Correo electronico base de datos")
    email_input.send_keys(Keys.ENTER)
    pausa_random()

    password_input = wait.until(EC.presence_of_element_located((By.NAME, "Passwd")))
    password_input.clear()
    password_input.send_keys("Contraseña para iniciar sesion")  
    password_input.send_keys(Keys.ENTER)
    pausa_random()

def ingresar_busqueda(driver, wait):
    print(" Ingresando cadena de búsqueda...")
    try:
        search_box = wait.until(EC.presence_of_element_located((By.ID, "qs")))
        query = '"generative artificial intelligence"'
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.ENTER)
        print(f" Búsqueda '{query}' ingresada.")
        pausa_random()
    except TimeoutException:
        print(" No se encontró la caja de búsqueda.")

def cambiar_resultados_por_pagina(driver, wait):
    print(" Cambiando a 100 resultados por página...")
    try:
        results_btn = wait.until(
            EC.element_to_be_clickable((By.XPATH, "//span[@class='anchor-text' and text()='100']"))
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", results_btn)
        pausa_random()
        driver.execute_script("arguments[0].click();", results_btn)
        pausa_random()
        print("100 resultados por página aplicados.")
    except TimeoutException:
        print("No se pudo cambiar la cantidad de resultados por página.")

def cerrar_popup(driver):
    try:
        no_account_btn = driver.find_element(By.XPATH, "//button[contains(., 'Usar Chrome sin una cuenta')]")
        no_account_btn.click()
        print(" Popup cerrado con 'Usar Chrome sin una cuenta'.")
    except NoSuchElementException:
        try:
            driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
            print(" Popup cerrado con ESC.")
        except:
            print("No apareció popup.")

# ==============================
# EXPORTAR RESULTADOS
# ==============================
def exportar_pagina(driver, wait, page):
    print(f"\n=== Procesando página {page} ===")
    try:
        # Seleccionar todos
        try:
            select_all_span = wait.until(
                EC.element_to_be_clickable((By.XPATH, "//label[@for='select-all-results']//span"))
            )
            driver.execute_script("arguments[0].scrollIntoView({block:'center'});", select_all_span)
            pausa_random()

            select_all_input = driver.find_element(By.ID, "select-all-results")
            if select_all_input.is_selected():
                driver.execute_script("arguments[0].click();", select_all_span)
                pausa_random()
            driver.execute_script("arguments[0].click();", select_all_span)
            print("Todos los resultados seleccionados.")
        except Exception:
            print("Usando checkboxes individuales...")
            checkboxes = wait.until(
                EC.presence_of_all_elements_located((By.XPATH, "//input[@type='checkbox' and contains(@class,'checkbox-input')]"))
            )
            for cb in checkboxes:
                if not cb.is_selected():
                    driver.execute_script("arguments[0].click();", cb)
            print("Todos los artículos seleccionados manualmente.")

        export_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(@class,'export-all-link-text')]")))
        driver.execute_script("arguments[0].click();", export_btn)
        pausa_random()

        bibtex_btn = wait.until(EC.element_to_be_clickable((By.XPATH, "//span[contains(., 'Export citation to BibTeX')]")))
        driver.execute_script("arguments[0].click();", bibtex_btn)
        print(" Exportación a BibTeX iniciada.")
        pausa_random()
    except Exception as e:
        print(f" Error exportando página {page}: {e}")

def ir_a_siguiente(driver):
    try:
        next_btn = driver.find_element(By.XPATH, "//a[.//span[text()='next']]")
        if "disabled" in next_btn.get_attribute("class").lower():
            print(" Última página alcanzada.")
            return False
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", next_btn)
        pausa_random()
        driver.execute_script("arguments[0].click();", next_btn)
        print("➡ Pasando a la siguiente página...")
        pausa_random()
        return True
    except NoSuchElementException:
        print(" Botón 'next' no encontrado. Fin del proceso.")
        return False

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    limpiar_perfil()
    copiar_perfil()
    driver = configurar_driver()
    wait = WebDriverWait(driver, 20)

    abrir_biblioteca(driver, wait)
    abrir_sciencedirect(driver)
    login_google(driver, wait)
    ingresar_busqueda(driver, wait)
    cerrar_popup(driver)

    cambiar_resultados_por_pagina(driver, wait)

    page = 1
    while page <= MAX_PAGES:
        exportar_pagina(driver, wait, page)
        if not ir_a_siguiente(driver):
            break
        page += 1

    print("\n Proceso de exportación finalizado.")
