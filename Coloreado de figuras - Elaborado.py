import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import os

class MatrizAImagen:
    def __init__(self):
        """
        Convertidor de matriz numérica a imagen colorida.
        Paletas simplificadas con colores básicos fundamentales.
        """
        # Definir paletas de colores básicas
        self.paletas = {
            "basicos": [
                "#FFFFFF",  # 0 - Blanco
                "#000000",  # 1 - Negro
                "#FF0000",  # 2 - Rojo (primario)
                "#FFFF00",  # 3 - Amarillo (primario)
                "#0000FF",  # 4 - Azul (primario)
                "#00FF00",  # 5 - Verde (secundario)
                "#FF8000",  # 6 - Naranja (secundario)
                "#8000FF",  # 7 - Violeta (secundario)
                "#FF0080",  # 8 - Rosa (terciario)
                "#80FF00"   # 9 - Lima (terciario)
            ],
            
            "primarios": [
                "#FFFFFF",  # 0 - Blanco
                "#000000",  # 1 - Negro
                "#FF0000",  # 2 - Rojo
                "#FFFF00",  # 3 - Amarillo
                "#0000FF",  # 4 - Azul
                "#808080",  # 5 - Gris
                "#C0C0C0",  # 6 - Gris claro
                "#404040",  # 7 - Gris oscuro
                "#800000",  # 8 - Rojo oscuro
                "#000080"   # 9 - Azul oscuro
            ],
            
            "secundarios": [
                "#FFFFFF",  # 0 - Blanco
                "#000000",  # 1 - Negro
                "#00FF00",  # 2 - Verde (amarillo + azul)
                "#FF8000",  # 3 - Naranja (rojo + amarillo)
                "#8000FF",  # 4 - Violeta (rojo + azul)
                "#80FF80",  # 5 - Verde claro
                "#FFBF80",  # 6 - Naranja claro
                "#BF80FF",  # 7 - Violeta claro
                "#008000",  # 8 - Verde oscuro
                "#804000"   # 9 - Naranja oscuro
            ],
            
            "rueda_color": [
                "#FFFFFF",  # 0 - Blanco
                "#000000",  # 1 - Negro
                "#FF0000",  # 2 - Rojo (primario)
                "#FF8000",  # 3 - Naranja (secundario)
                "#FFFF00",  # 4 - Amarillo (primario)
                "#80FF00",  # 5 - Lima (terciario)
                "#00FF00",  # 6 - Verde (secundario)
                "#00FF80",  # 7 - Turquesa (terciario)
                "#0000FF",  # 8 - Azul (primario)
                "#8000FF"   # 9 - Violeta (secundario)
            ],
            
            "calidos": [
                "#FFFFFF",  # 0 - Blanco
                "#000000",  # 1 - Negro
                "#FF0000",  # 2 - Rojo
                "#FF4000",  # 3 - Rojo-naranja
                "#FF8000",  # 4 - Naranja
                "#FFBF00",  # 5 - Naranja-amarillo
                "#FFFF00",  # 6 - Amarillo
                "#BFFF00",  # 7 - Amarillo-verde
                "#FF0040",  # 8 - Rosa-rojo
                "#FF8040"   # 9 - Coral
            ],
            
            "frios": [
                "#FFFFFF",  # 0 - Blanco
                "#000000",  # 1 - Negro
                "#0000FF",  # 2 - Azul
                "#0040FF",  # 3 - Azul-violeta
                "#8000FF",  # 4 - Violeta
                "#4000FF",  # 5 - Índigo
                "#00FF00",  # 6 - Verde
                "#0080FF",  # 7 - Azul claro
                "#0040BF",  # 8 - Azul medio
                "#004080"   # 9 - Azul oscuro
            ]
        }
    
    def mostrar_paletas_disponibles(self):
        """Muestra todas las paletas de colores disponibles"""
        print("PALETAS DE COLORES BÁSICOS DISPONIBLES:")
        print("="*70)
        
        descripcion_paletas = {
            "basicos": "Colores primarios, secundarios y terciarios básicos",
            "primarios": "Enfoque en colores primarios (rojo, amarillo, azul)",
            "secundarios": "Enfoque en colores secundarios (verde, naranja, violeta)", 
            "rueda_color": "Rueda de color completa ordenada",
            "calidos": "Colores cálidos (rojos, naranjas, amarillos)",
            "frios": "Colores fríos (azules, verdes, violetas)"
        }
        
        for nombre, colores in self.paletas.items():
            print(f"\n{nombre.upper()}:")
            print(f"   {descripcion_paletas[nombre]}")
            for i, color in enumerate(colores):
                tipo_color = self._clasificar_color(i, nombre)
                print(f"  {i}: {color} ({tipo_color})")
        
        print(f"\n{'='*70}")
        print("Usa el nombre de la paleta en la función convertir_matriz()")
        print("Ejemplo: convertir_matriz(matriz, paleta='basicos')")
    
    def _clasificar_color(self, indice, paleta):
        """Clasifica el tipo de color según su índice y paleta"""
        if indice == 0:
            return "Blanco"
        elif indice == 1:
            return "Negro"
        elif paleta == "primarios" and indice in [2, 3, 4]:
            return "Primario"
        elif paleta == "secundarios" and indice in [2, 3, 4]:
            return "Secundario"
        elif paleta == "rueda_color":
            tipos = ["", "", "Primario", "Secundario", "Primario", "Terciario", 
                    "Secundario", "Terciario", "Primario", "Secundario"]
            return tipos[indice] if indice < len(tipos) else "Color"
        else:
            return "Color"
    
    def parsear_matriz(self, texto_matriz):
        """
        Convierte texto de matriz copiado en array numpy.
        Acepta múltiples formatos de entrada.
        """
        # Limpiar el texto
        texto = texto_matriz.strip()
        
        # Caso 1: Matriz con corchetes por fila
        if '[' in texto and ']' in texto:
            filas = []
            for linea in texto.split('\n'):
                linea = linea.strip()
                if '[' in linea and ']' in linea:
                    # Extraer contenido entre corchetes
                    contenido = linea[linea.find('[')+1:linea.find(']')]
                    # Convertir a números
                    numeros = [int(x.strip()) for x in contenido.split(',') if x.strip().isdigit()]
                    if numeros:
                        filas.append(numeros)
            
            if filas:
                return np.array(filas)
        filas = []
        for linea in texto.split('\n'):
            linea = linea.strip()
            if not linea:
                continue
            
            # Intentar separar por comas primero, luego por espacios
            if ',' in linea:
                numeros = [int(x.strip()) for x in linea.split(',') if x.strip().isdigit()]
            else:
                numeros = [int(x) for x in linea.split() if x.isdigit()]
            
            if numeros:
                filas.append(numeros)
        
        if filas:
            return np.array(filas)
        
        raise ValueError("No se pudo parsear la matriz. Verifica el formato.")
    
    def convertir_matriz(self, matriz_texto, 
                        paleta="basicos", 
                        tamaño_pixel=50,
                        mostrar_numeros=True,
                        guardar_como=None,
                        mostrar_imagen=True):
        """
        Convierte matriz de texto a imagen colorida.
        
        Parámetros:
        • matriz_texto: Texto de la matriz copiada
        • paleta: Nombre de la paleta de colores
        • tamaño_pixel: Tamaño de cada píxel en la imagen final
        • mostrar_numeros: Si mostrar números en cada celda
        • guardar_como: Ruta para guardar la imagen (opcional)
        • mostrar_imagen: Si mostrar la imagen en pantalla
        """
        
        # Parsear la matriz
        try:
            matriz = self.parsear_matriz(matriz_texto)
            print(f"Matriz parseada correctamente: {matriz.shape}")
        except Exception as e:
            print(f"Error al parsear matriz: {e}")
            return None
        
        # Verificar paleta
        if paleta not in self.paletas:
            print(f"Paleta '{paleta}' no encontrada. Usando 'basicos'")
            paleta = "basicos"
        
        colores = self.paletas[paleta]
        print(f"Usando paleta: {paleta}")
        
        # Obtener valores únicos en la matriz
        valores_unicos = np.unique(matriz)
        print(f"Valores en la matriz: {valores_unicos}")
        
        # Verificar si hay suficientes colores
        if len(valores_unicos) > len(colores):
            print(f" Advertencia: La matriz tiene {len(valores_unicos)} valores únicos")
            print(f"   pero la paleta solo tiene {len(colores)} colores.")
            print("   Los valores altos usarán el último color.")
        
        # Crear la visualización
        filas, columnas = matriz.shape
        fig, ax = plt.subplots(figsize=(columnas * 0.8, filas * 0.8))
        
        # Dibujar cada celda
        for i in range(filas):
            for j in range(columnas):
                valor = matriz[i, j]
                
                # Obtener color (si valor excede colores disponibles, usar el último)
                indice_color = min(valor, len(colores) - 1)
                color = colores[indice_color]
                
                # Dibujar rectángulo
                rect = plt.Rectangle((j, filas-1-i), 1, 1,
                                   facecolor=color,
                                   edgecolor='black',
                                   linewidth=1)
                ax.add_patch(rect)
                
                # Añadir número si se solicita
                if mostrar_numeros:
                    # Determinar color del texto para contraste
                    color_texto = self._obtener_color_contraste(color)
                    
                    ax.text(j+0.5, filas-1-i+0.5, str(valor),
                           ha='center', va='center',
                           fontsize=max(8, 20 - max(filas, columnas)),
                           color=color_texto,
                           fontweight='bold')
        
        # Configurar el gráfico
        ax.set_xlim(0, columnas)
        ax.set_ylim(0, filas)
        ax.set_aspect('equal')
        ax.set_xticks(range(columnas+1))
        ax.set_yticks(range(filas+1))
        ax.grid(True, alpha=0.3)
        
        # Título y información
        titulo = f"Matriz Colorizada - Paleta: {paleta.title()}\n"
        titulo += f"Tamaño: {filas}x{columnas} | Valores: {valores_unicos}"
        ax.set_title(titulo, fontsize=12, fontweight='bold', pad=20)
        
        # Crear leyenda de colores
        self._crear_leyenda(ax, valores_unicos, colores, paleta)
        
        plt.tight_layout()
        
        # Guardar si se especifica
        if guardar_como:
            try:
                # Crear directorio si no existe
                directorio = os.path.dirname(guardar_como)
                if directorio and not os.path.exists(directorio):
                    os.makedirs(directorio)
                
                plt.savefig(guardar_como, dpi=300, bbox_inches='tight')
                print(f"Imagen guardada como: {guardar_como}")
            except Exception as e:
                print(f"Error al guardar: {e}")
        
        # Mostrar imagen
        if mostrar_imagen:
            plt.show()
        
        return fig, matriz
    
    def _obtener_color_contraste(self, color_hex):
        """Determina si usar texto blanco o negro para contraste"""
        # Convertir hex a RGB
        color_hex = color_hex.lstrip('#')
        r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        
        # Calcular luminancia
        luminancia = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        
        # Retornar blanco para colores oscuros, negro para claros
        return 'white' if luminancia < 0.5 else 'black'
    
    def _crear_leyenda(self, ax, valores, colores, paleta):
        """Crea una leyenda de colores"""
        from matplotlib.patches import Rectangle
        
        # Crear leyenda fuera del gráfico
        leyenda_elementos = []
        for valor in valores:
            indice_color = min(valor, len(colores) - 1)
            color = colores[indice_color]
            tipo = self._clasificar_color(indice_color, paleta)
            leyenda_elementos.append(Rectangle((0,0),1,1, facecolor=color, 
                                             edgecolor='black', 
                                             label=f'{valor}: {tipo}'))
        
        ax.legend(handles=leyenda_elementos, 
                 loc='center left', bbox_to_anchor=(1, 0.5),
                 title=f'Leyenda - {paleta.title()}')

# ================================
# EJECUCIÓN DIRECTA CON MATRIZ EJEMPLO
# ================================

if __name__ == "__main__":
    print("CONVERTIDOR DE MATRIZ A IMAGEN - GENERANDO EJEMPLO")
    print("="*70)
    
    # Crear el conversor
    conversor = MatrizAImagen()
    
    # Matriz ejemplo activa
    tu_matriz = """
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    [0, 0, 0, 1, 1, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    """
    
    print("Generando imagen de ejemplo con paleta 'basicos'...")
    
    # Convertir la matriz ejemplo
    conversor.convertir_matriz(
        tu_matriz,
        paleta="basicos",  # Cambia aquí por: primarios, secundarios, rueda_color, calidos, frios
        guardar_como="matriz_ejemplo_colorida.png",
        mostrar_imagen=True
    )