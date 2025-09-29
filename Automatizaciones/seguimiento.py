import pandas as pd
import time
import matplotlib.pyplot as plt
from collections import Counter
import os
import re
import traceback

# ==============================
# CONFIGURACIÓN
# ==============================
CARPETA_SALIDA = r"C:\Users\clare\bibliometria\DataFinal"
os.makedirs(CARPETA_SALIDA, exist_ok=True)
RUTA = os.path.join(CARPETA_SALIDA, "unificado.csv")
MAX_HOLES = 200000  # umbral para pigeonhole

# ==============================
# 1. Lectura base del CSV
# ==============================
def cargar_datos():
    df = pd.read_csv(RUTA, sep=";", encoding="latin1", engine="python", on_bad_lines="skip")
    df.fillna("Sin información", inplace=True)
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").fillna(0).astype(int)
    return df

df = cargar_datos()
n = len(df)
print(f"Número de registros detectados: {n}")

# ==============================
# 2. Guardar productos ordenados (TimSort)
# ==============================
df_ordenado = df.sort_values(by=["Year", "Title"], ascending=[True, True])
ruta_productos = os.path.join(CARPETA_SALIDA, "productos_ordenados.csv")
df_ordenado.to_csv(ruta_productos, sep=";", index=False, encoding="utf-8-sig")
print(f"✅ Productos ordenados guardados en: {ruta_productos}")

# ==============================
# 3. Algoritmos de ordenamiento
# ==============================
def timsort(arr): return sorted(arr)
def comb_sort(arr):
    gap = len(arr); shrink = 1.3; sorted_ = False
    while not sorted_:
        gap = int(gap / shrink)
        if gap <= 1: gap = 1; sorted_ = True
        i = 0
        while i + gap < len(arr):
            if arr[i] > arr[i + gap]:
                arr[i], arr[i+gap] = arr[i+gap], arr[i]; sorted_ = False
            i += 1
    return arr
def selection_sort(arr):
    for i in range(len(arr)):
        min_idx = i
        for j in range(i+1, len(arr)):
            if arr[j] < arr[min_idx]: min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
    return arr
class TreeNode: 
    def __init__(self,val): self.val=val; self.left=None; self.right=None
def insert(root,val):
    if root is None: return TreeNode(val)
    if val<root.val: root.left=insert(root.left,val)
    else: root.right=insert(root.right,val)
    return root
def inorder(root,res):
    if root: inorder(root.left,res); res.append(root.val); inorder(root.right,res)
def tree_sort(arr):
    if not arr: return arr
    root=TreeNode(arr[0])
    for v in arr[1:]: insert(root,v)
    res=[]; inorder(root,res); return res
def pigeonhole_sort(arr):
    if len(arr)==0: return arr
    mn,mx=min(arr),max(arr); size=mx-mn+1
    if size>MAX_HOLES: raise MemoryError(f"Demasiados huecos: {size}")
    holes=[[] for _ in range(size)]
    for x in arr: holes[x-mn].append(x)
    res=[]; [res.extend(h) for h in holes]; return res
def bucket_sort(arr):
    if len(arr)==0: return arr
    mn,mx=min(arr),max(arr); denom=(mx-mn+1)
    if denom==0: return arr
    buckets=[[] for _ in range(10)]
    for x in arr: buckets[int((x-mn)/denom*9)].append(x)
    res=[]; [res.extend(sorted(b)) for b in buckets]; return res
def quicksort(arr):
    if len(arr)<=1: return arr
    p=arr[len(arr)//2]
    left=[x for x in arr if x<p]; mid=[x for x in arr if x==p]; right=[x for x in arr if x>p]
    return quicksort(left)+mid+quicksort(right)
def heapsort(arr):
    import heapq; arr=arr[:]; heapq.heapify(arr); return [heapq.heappop(arr) for _ in range(len(arr))]
def bitonic_sort(arr): return sorted(arr)
def gnome_sort(arr):
    i=0; arr=arr[:]
    while i<len(arr):
        if i==0 or arr[i]>=arr[i-1]: i+=1
        else: arr[i],arr[i-1]=arr[i-1],arr[i]; i-=1
    return arr
def binary_insertion_sort(arr):
    arr=arr[:]
    for i in range(1,len(arr)):
        key=arr[i]; l,r=0,i-1
        while l<=r:
            m=(l+r)//2
            if arr[m]>key: r=m-1
            else: l=m+1
        for j in range(i,l,-1): arr[j]=arr[j-1]
        arr[l]=key
    return arr
def counting_sort_for_radix(arr,exp):
    n=len(arr); output=[0]*n; count=[0]*10
    for i in range(n): count[(arr[i]//exp)%10]+=1
    for i in range(1,10): count[i]+=count[i-1]
    for i in range(n-1,-1,-1):
        index=(arr[i]//exp)%10; output[count[index]-1]=arr[i]; count[index]-=1
    for i in range(n): arr[i]=output[i]
def radix_sort(arr):
    if len(arr)==0: return arr
    mn=min(arr)
    if mn<0: arr=[x-mn for x in arr]
    mx=max(arr); exp=1
    while mx//exp>0: counting_sort_for_radix(arr,exp); exp*=10
    if mn<0: arr=[x+mn for x in arr]; return arr

# ==============================
# 4. Ejecución: cada algoritmo recarga CSV
# ==============================
algorithms={
    "TimSort":timsort,"Comb Sort":comb_sort,"Selection Sort":selection_sort,"Tree Sort":tree_sort,
    "QuickSort":quicksort,"HeapSort":heapsort,"Bitonic Sort":bitonic_sort,"Gnome Sort":gnome_sort,
    "Binary Insertion Sort":binary_insertion_sort,"Pigeonhole Sort":pigeonhole_sort,
    "BucketSort":bucket_sort,"RadixSort":radix_sort
}
numeric_only={"Pigeonhole Sort","BucketSort","RadixSort"}
tiempos=[]

for nombre,funcion in algorithms.items():
    df = cargar_datos()
    datos_nums = df["Year"].astype(int).tolist()
    datos_tuples = list(zip(df["Year"].astype(int).tolist(), df["Title"].astype(str).tolist()))
    entrada = datos_nums[:] if nombre in numeric_only else datos_tuples[:]
    start=time.perf_counter()
    try:
        funcion(entrada)
        elapsed=time.perf_counter()-start
        print(f"{nombre}: {elapsed:.6f} s")
    except MemoryError as me:
        elapsed=None; print(f"{nombre}: Memoria insuficiente ({me})")
    except Exception:
        elapsed=None; print(f"{nombre}: Error"); traceback.print_exc()
    tiempos.append({"Algoritmo":nombre,"Tamaño":n,"Tiempo":elapsed})

# ==============================
# 5. Guardar tiempos + gráfico (barras verticales)
# ==============================
df_tiempos = pd.DataFrame(tiempos)
ruta_tiempos = os.path.join(CARPETA_SALIDA, "tiempos_reales.csv")
df_tiempos.to_csv(ruta_tiempos, sep=";", index=False, encoding="utf-8-sig")

# Filtrar y ordenar por tiempo ascendente
df_plot = df_tiempos[df_tiempos["Tiempo"].notna()].sort_values("Tiempo")

# Gráfico de barras verticales
plt.figure(figsize=(10, 6))
bars = plt.bar(df_plot["Algoritmo"], df_plot["Tiempo"])
plt.ylabel("Tiempo (s)")
plt.title(f"Tiempos de ordenamiento (n={n})")

# Etiquetas con tiempo encima de cada barra
max_time = df_plot["Tiempo"].max() if not df_plot.empty else 0
pad = max_time * 0.02 if max_time > 0 else 0.001
for b in bars:
    h = b.get_height()
    plt.text(b.get_x() + b.get_width() / 2, h + pad, f"{h:.6f}", 
             ha="center", va="bottom", fontsize=8)

# Ajuste para evitar que se superpongan etiquetas
plt.xticks(rotation=20, ha="right")
plt.tight_layout()

# Guardar y mostrar
ruta_grafico = os.path.join(CARPETA_SALIDA, "grafico_tiempos_reales.png")
plt.savefig(ruta_grafico)
plt.show()

# ==============================
# 6. Top 15 autores
# ==============================
def extraer_autores(cadena):
    if not isinstance(cadena,str) or cadena.strip().lower() in ("sin información","desconocido",""):
        return []
    partes=re.split(r"\s+and\s+",cadena); autores=[]
    for p in partes:
        p=p.strip()
        if ";" in p:
            a,n=p.split(";",1); autores.append(f"{a.strip()} {n.strip()}")
        else: autores.append(p)
    return autores

todos=[]; [todos.extend(extraer_autores(f)) for f in df["Authors"].astype(str)]
conteo=Counter(todos); top15=conteo.most_common(15)
df_top15=pd.DataFrame(top15,columns=["Autor","Apariciones"]).sort_values("Apariciones")
ruta_top15=os.path.join(CARPETA_SALIDA,"top15_autores_completo.csv")
df_top15.to_csv(ruta_top15,sep=";",index=False,encoding="utf-8-sig")

plt.figure(figsize=(8,6))
bars=plt.barh(df_top15["Autor"],df_top15["Apariciones"],color="lightgreen")
plt.xlabel("Apariciones"); plt.title("Top 15 Autores (Nombre Completo)")
for b in bars: w=b.get_width(); plt.text(w+0.5,b.get_y()+b.get_height()/2,str(int(w)),va="center")
plt.tight_layout()
ruta_grafico_top15=os.path.join(CARPETA_SALIDA,"grafico_top15_autores.png")
plt.savefig(ruta_grafico_top15); plt.show()
