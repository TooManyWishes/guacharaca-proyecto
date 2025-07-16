from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def detectar_figura_optimizado(ruta_imagen: str,
                              tamaño: int = 15,
                              metodo: str = "contraste_mejorado",
                              umbral_blanco: int = 240,
                              sensibilidad: float = 0.8) -> np.ndarray:
    """
    Detecta la figura principal en una imagen con fondo blanco.
    Sensibilidad ajustable.
    
    Parámetros:
    • ruta_imagen    : path de la imagen
    • tamaño         : matriz objetivo (15x15, 20x20, etc.) - esto es configurable casi al final del código, revisar los comentarios para guiarse
    • metodo         : método de detección
    • umbral_blanco  : threshold para considerar un píxel como fondo blanco
    • sensibilidad   : qué tan estricto ser con la detección (0.1 a 1.0)
    
    Da de regreso: matriz con 0 (fondo) y 1 (figura)
    """
    
    # Cargar y procesar imagen
    img = Image.open(ruta_imagen).convert("RGB")
    img = img.resize((tamaño, tamaño), Image.Resampling.LANCZOS)
    pix = np.asarray(img, dtype=np.uint8)
    
    if metodo == "contraste_mejorado":
        return _detectar_por_contraste_mejorado(pix, umbral_blanco, sensibilidad)
    elif metodo == "diferencia_adaptativa":
        return _detectar_por_diferencia_adaptativa(pix, umbral_blanco, sensibilidad)
    elif metodo == "luminancia_precisa":
        return _detectar_por_luminancia_precisa(pix, umbral_blanco, sensibilidad)
    elif metodo == "bordes_combinados":
        return _detectar_por_bordes_combinados(pix, umbral_blanco, sensibilidad)
    else:
        return _detectar_basico(pix, umbral_blanco)

def _detectar_basico(pix, umbral_blanco):
    """Método básico: simplemente busca píxeles no blancos"""
    fondo = (pix[:, :, 0] >= umbral_blanco) & \
            (pix[:, :, 1] >= umbral_blanco) & \
            (pix[:, :, 2] >= umbral_blanco)
    return (~fondo).astype(int)

def _detectar_por_contraste_mejorado(pix, umbral_blanco, sensibilidad):
    """Detecta basado en contraste local mejorado"""
    filas, columnas = pix.shape[:2]
    matriz = np.zeros((filas, columnas), dtype=int)
    
    # Calcular contraste local en ventanas variables
    for i in range(filas):
        for j in range(columnas):
            # Ventana adaptativa (más grande en bordes)
            ventana_size = 2 if min(i, j, filas-1-i, columnas-1-j) > 1 else 1
            
            i_min = max(0, i - ventana_size)
            i_max = min(filas, i + ventana_size + 1)
            j_min = max(0, j - ventana_size)
            j_max = min(columnas, j + ventana_size + 1)
            
            ventana = pix[i_min:i_max, j_min:j_max]
            
            # Múltiples métricas de contraste
            std_r = np.std(ventana[:, :, 0])
            std_g = np.std(ventana[:, :, 1])
            std_b = np.std(ventana[:, :, 2])
            std_promedio = (std_r + std_g + std_b) / 3
            
            # Diferencia con esquinas (probablemente fondo)
            esquinas = [
                ventana[0, 0], ventana[0, -1], 
                ventana[-1, 0], ventana[-1, -1]
            ]
            promedio_esquinas = np.mean(esquinas, axis=0)
            pixel_actual = pix[i, j]
            diferencia_esquinas = np.mean(np.abs(pixel_actual - promedio_esquinas))
            
            # Combinación de criterios
            umbral_contraste = 15 * sensibilidad
            umbral_diferencia = 20 * sensibilidad
            
            if std_promedio > umbral_contraste or diferencia_esquinas > umbral_diferencia:
                # Verificar que no sea muy blanco
                if not (pixel_actual[0] >= umbral_blanco and 
                       pixel_actual[1] >= umbral_blanco and 
                       pixel_actual[2] >= umbral_blanco):
                    matriz[i, j] = 1
    
    return matriz

def _detectar_por_diferencia_adaptativa(pix, umbral_blanco, sensibilidad):
    """Detecta comparando cada píxel con el fondo blanco esperado"""
    filas, columnas = pix.shape[:2]
    
    # Estimar color de fondo desde las esquinas
    esquinas = [
        pix[0, 0], pix[0, -1], pix[-1, 0], pix[-1, -1],
        pix[0, columnas//2], pix[filas//2, 0], 
        pix[filas//2, -1], pix[-1, columnas//2]
    ]
    
    # Filtrar solo esquinas que parecen blancas
    esquinas_blancas = [e for e in esquinas if np.all(e >= umbral_blanco - 20)]
    
    if esquinas_blancas:
        color_fondo = np.mean(esquinas_blancas, axis=0)
    else:
        color_fondo = np.array([255, 255, 255])  # Asumir blanco puro
    
    # Calcular diferencia con el fondo
    diferencia = np.sqrt(np.sum((pix - color_fondo)**2, axis=2))
    
    # Umbral adaptativo basado en sensibilidad
    umbral_diferencia = 30 * sensibilidad
    
    matriz = (diferencia > umbral_diferencia).astype(int)
    return matriz

def _detectar_por_luminancia_precisa(pix, umbral_blanco, sensibilidad):
    """Detecta usando luminancia con umbral adaptativo"""
    # Calcular luminancia (percepción humana del brillo)
    luminancia = 0.299 * pix[:, :, 0] + 0.587 * pix[:, :, 1] + 0.114 * pix[:, :, 2]
    
    # Estimar luminancia del fondo
    bordes = np.concatenate([
        luminancia[0, :], luminancia[-1, :], 
        luminancia[:, 0], luminancia[:, -1]
    ])
    luminancia_fondo = np.median(bordes[bordes >= umbral_blanco * 0.9])
    
    # Umbral adaptativo
    umbral_luminancia = luminancia_fondo - (50 * sensibilidad)
    
    matriz = (luminancia < umbral_luminancia).astype(int)
    return matriz

def _detectar_por_bordes_combinados(pix, umbral_blanco, sensibilidad):
    """Combina detección de bordes con análisis de color"""
    from scipy import ndimage
    
    # Convertir a escala de grises
    gris = np.mean(pix, axis=2)
    
    # Detectar bordes con Sobel
    bordes_x = ndimage.sobel(gris, axis=1)
    bordes_y = ndimage.sobel(gris, axis=0)
    bordes = np.sqrt(bordes_x**2 + bordes_y**2)
    
    # Normalizar bordes
    if np.max(bordes) > 0:
        bordes = bordes / np.max(bordes)
    
    # Detección básica de no-blancos
    no_blanco = ~((pix[:, :, 0] >= umbral_blanco) & 
                  (pix[:, :, 1] >= umbral_blanco) & 
                  (pix[:, :, 2] >= umbral_blanco))
    
    # Combinar bordes y no-blancos
    umbral_borde = 0.1 * sensibilidad
    matriz = ((bordes > umbral_borde) | no_blanco).astype(int)
    
    return matriz

def mostrar_resultado_simple(matriz: np.ndarray, 
                           titulo: str = "Detección de Figura",
                           ruta_imagen: str = ""):
    """Muestra el resultado en terminal y ventana gráfica simple"""
    
    filas, columnas = matriz.shape
    
    # === MOSTRAR EN TERMINAL ===
    print(f"\n{'='*60}")
    print(f"{titulo.center(60)}")
    print(f"{'='*60}")
    print(f"Imagen: {ruta_imagen.split('/')[-1] if ruta_imagen else 'N/A'}")
    print(f"Tamaño: {filas}x{columnas}")
    print(f"Píxeles de figura detectados: {np.sum(matriz)}")
    print(f"Porcentaje de cobertura: {(np.sum(matriz)/(filas*columnas)*100):.1f}%")
    print(f"{'='*60}")
    
    # === MOSTRAR MATRIZ EN FORMATO TÍPICO ===
    print("MATRIZ RESULTADO:")
    for i in range(filas):
        fila_str = "["
        for j in range(columnas):
            if j > 0:
                fila_str += ", "
            fila_str += str(matriz[i, j])
        fila_str += "]"
        print(fila_str)
    print(f"{'='*60}")
    
    # === MOSTRAR EN VENTANA GRÁFICA ===
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Crear visualización
    for i in range(filas):
        for j in range(columnas):
            valor = matriz[i, j]
            color = "#000000" if valor == 1 else "#FFFFFF"  # Negro o blanco
            
            rect = plt.Rectangle((j, filas-1-i), 1, 1, 
                               facecolor=color, 
                               edgecolor="#CCCCCC", 
                               linewidth=1)
            ax.add_patch(rect)
            
            # Añadir número
            text_color = "white" if valor == 1 else "black"
            ax.text(j+0.5, filas-1-i+0.5, str(valor),
                   ha="center", va="center", 
                   fontsize=10, color=text_color, 
                   fontweight="bold")
    
    # Configurar gráfico
    ax.set_xlim(0, columnas)
    ax.set_ylim(0, filas)
    ax.set_xticks(range(columnas+1))
    ax.set_yticks(range(filas+1))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.5)
    
    # Título
    imagen_nombre = ruta_imagen.split('/')[-1] if ruta_imagen else "Imagen"
    ax.set_title(f"{titulo}\n{imagen_nombre} ({filas}x{columnas})", 
                fontsize=12, fontweight="bold", pad=20)
    
    # Estadísticas en el gráfico
    stats_text = f"Píxeles figura: {np.sum(matriz)}\nCobertura: {(np.sum(matriz)/(filas*columnas)*100):.1f}%"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def procesar_imagen_simple(ruta_imagen: str, 
                          tamaño: int = 15,
                          metodo: str = "contraste_mejorado",
                          sensibilidad: float = 0.8,
                          mostrar_todos_metodos: bool = False,
                          ruta_guardado: str = None):
    """
    Función principal simplificada para detección de figura.
    
    Nuevo parámetro:
    • ruta_guardado: Ruta donde se guardará la imagen resultado. Si es None, se guarda en la carpeta actual.
    """
    
    print(f"Procesando: {ruta_imagen}")
    print(f"Tamaño: {tamaño}x{tamaño}")
    print(f"Método: {metodo}")
    print(f"Sensibilidad: {sensibilidad}")
    
    if mostrar_todos_metodos:
        metodos = ["contraste_mejorado", "diferencia_adaptativa", 
                  "luminancia_precisa", "bordes_combinados"]
        
        for metodo_actual in metodos:
            try:
                print(f"\n🔍 Probando método: {metodo_actual}")
                matriz = detectar_figura_optimizado(ruta_imagen, tamaño, 
                                                  metodo_actual, sensibilidad=sensibilidad)
                fig = mostrar_resultado_simple(matriz, 
                                             f"Método: {metodo_actual}", 
                                             ruta_imagen)
                
                # Guardar con ruta personalizada
                if ruta_guardado:
                    # Crear directorio si no existe
                    os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
                    nombre_archivo = os.path.join(ruta_guardado, f"deteccion_{metodo_actual}_{tamaño}x{tamaño}.png")
                else:
                    nombre_archivo = f"deteccion_{metodo_actual}_{tamaño}x{tamaño}.png"
                
                fig.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
                print(f"Guardado como: {nombre_archivo}")
                
                plt.show()
                
                input("Presiona Enter para el siguiente método...")
                plt.close(fig)
                
            except Exception as e:
                print(f"Error con {metodo_actual}: {e}")
    else:
        try:
            matriz = detectar_figura_optimizado(ruta_imagen, tamaño, metodo, 
                                              sensibilidad=sensibilidad)
            fig = mostrar_resultado_simple(matriz, f"Detección - {metodo}", ruta_imagen)
            
            # Guardar resultado con ruta personalizada
            if ruta_guardado:
                # Crear directorio si no existe
                os.makedirs(os.path.dirname(ruta_guardado), exist_ok=True)
                
                # Si ruta_guardado es una carpeta, crear nombre de archivo
                if os.path.isdir(ruta_guardado) or ruta_guardado.endswith('/') or ruta_guardado.endswith('\\'):
                    nombre_archivo = os.path.join(ruta_guardado, f"deteccion_{metodo}_{tamaño}x{tamaño}.png")
                else:
                    # Si ruta_guardado incluye el nombre del archivo
                    nombre_archivo = ruta_guardado
                    # Asegurar que tenga extensión .png
                    if not nombre_archivo.lower().endswith('.png'):
                        nombre_archivo += '.png'
            else:
                # Guardar en la carpeta actual
                nombre_archivo = f"deteccion_{metodo}_{tamaño}x{tamaño}.png"
            
            fig.savefig(nombre_archivo, dpi=300, bbox_inches='tight')
            print(f"Guardado como: {nombre_archivo}")
            
            plt.show()
            return matriz
            
        except Exception as e:
            print(f"Error: {e}")
            return None

# === EJEMPLO DE USO ===
if __name__ == "__main__":
    # Tu imagen
    ruta_imagen = "C:\\Users\\Sam.LAPTOPHP-2021\\Pictures\\Camera Roll\Gato.jpeg" # Para ingresar la ruta, considera que cada \ va 2 veces: \\ 
    # Ejemplo: C:\Users\Sam.LAPTOPHP-2021\Pictures\Camera Roll\Tambor.png - Ruta original.
    # C:\\Users\\Sam.LAPTOPHP-2021\\Pictures\\Camera Roll\Gato.jpeg
    # C:\\Users\\Sam.LAPTOPHP-2021\\Pictures\\Camera Roll\\Tambor.png - Ruta modificada para permitir la lectura en Phyton.
    
    # NUEVA FUNCIONALIDAD: Establecer ruta de guardado
    # Opción 1: Especificar carpeta donde guardar
    ruta_guardado = "C:\\Users\\Sam.LAPTOPHP-2021\\Desktop\\Resultados\\"
    
    # Opción 2: Especificar ruta completa con nombre de archivo
    # ruta_guardado = "C:\\Users\\Sam.LAPTOPHP-2021\\Desktop\\mi_deteccion_personalizada.png"
    
    # Opción 3: Dejar None para guardar en la carpeta actual
    # ruta_guardado = None
    
    print("DETECTOR DE FIGURAS OPTIMIZADO")
    print("=" * 50)
    
    # Opción 1: Usar el mejor método directamente
    matriz_resultado = procesar_imagen_simple(
        ruta_imagen=ruta_imagen,
        tamaño=20,  # Cambiar aquí el tamaño (rango máximo sugerido 15 - 25)
        metodo="contraste_mejorado",  # El que mejor funciona según tú
        sensibilidad=0.4,  # Ajustar entre 0.1 (menos estricto) y 1.0 (más estricto)
        ruta_guardado=ruta_guardado  # Establecer dónde guardar la imagen
    )
    
    # Opción 2: Probar todos los métodos para comparar
    # Descomenta las siguientes líneas si quieres probar todos:
    
    # print("\n¿Quieres probar todos los métodos? (y/n)")
    # if input().lower() == 'y':
    #     procesar_imagen_simple(ruta_imagen, 15, mostrar_todos_metodos=True, ruta_guardado=ruta_guardado)