# Spec — Utilidad de resolución de rutas absolutas (`src/utils.py`)

- **Tarea atómica:** `abs_dir_func`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** desarrollador del proyecto,
> **quiero** un módulo `src/utils.py` con funciones generales y, como primera
> función, una que convierta una ruta relativa en su ruta absoluta,
> **para** poder acceder de forma estable y consistente a los archivos del
> proyecto (empezando por los archivos de configuración) sin depender del
> directorio de trabajo desde el que se ejecute el código.

---

## 2. Motivación (por qué)

- La constitución exige que las rutas de datos sean **relativas** y se accedan de
  forma controlada. Una utilidad central de resolución evita rutas absolutas
  "hardcodeadas" y comportamientos distintos según el `cwd`.
- Centralizar utilidades comunes en `src/utils.py` evita duplicación a medida que
  crece el proyecto.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Crear el módulo `src/utils.py` como contenedor de **funciones generales**.
- Implementar la **primera función**: dada una ruta relativa, devolver su ruta
  absoluta.

### 3.2 Fuera de alcance

- Cualquier otra utilidad futura de `utils.py`.
- La carga o el parseo del contenido de los archivos de configuración (esta
  función solo resuelve rutas, no lee archivos).

---

## 4. Comportamiento esperado

- **Entrada:** una ruta relativa en forma de `str`.
- **Salida:** la ruta absoluta correspondiente como objeto `pathlib.Path`.
- La ruta relativa se interpreta respecto a la **raíz del proyecto**, de modo que
  el resultado sea el mismo sin importar desde dónde se ejecute el código.
- La función **verifica la existencia** de la ruta resuelta: si la ruta no
  existe, lanza `FileNotFoundError` y **detiene el proceso**.
- **Manejo de errores:** entrada inválida (no `str`, cadena vacía, o ruta
  absoluta en lugar de relativa) lanza `ValueError`; ruta resuelta inexistente
  lanza `FileNotFoundError`. Ambos errores detienen el proceso.
- **Caso de uso principal:** resolver rutas de **archivos de configuración**
  (por ejemplo, el archivo indicado por `CONFIG_FILENAME` en `.env`, ubicado en
  `configs/`).

---

## 5. Criterios de aceptación

1. **AC-1 — Módulo presente:** existe `src/utils.py`.
2. **AC-2 — Firma:** la función recibe un `str` (ruta relativa) y devuelve un
   `pathlib.Path`.
3. **AC-3 — Absoluta:** el `Path` devuelto es absoluto.
4. **AC-4 — Estable:** el resultado es el mismo independientemente del directorio
   de trabajo desde el que se invoque (se resuelve respecto a la raíz del
   proyecto).
5. **AC-5 — Verifica existencia:** la función comprueba que la ruta resuelta
   exista; si no existe, lanza `FileNotFoundError` y detiene el proceso. La
   entrada inválida (no `str`, vacía o ruta absoluta) lanza `ValueError`.
6. **AC-6 — Validación manual:** se demuestra de forma **exploratoria**
   (notebook o script suelto) resolviendo rutas de archivos de configuración del
   proyecto e inspeccionando que el `Path` absoluto resultante sea el correcto.

---

## 6. Supuestos y notas

- La constitución indica que las rutas se acceden vía configuración; esta función
  es precisamente la pieza base que habilita ese acceso estable.
- Esta especificación **no** define el *cómo* técnico (firma exacta, mecanismo de
  resolución de la raíz, manejo de errores); eso corresponde al `plan.md` de esta
  misma carpeta.

---

## 7. Siguientes pasos (metodología)

1. Elaborar `plan.md` con el detalle técnico de implementación.
2. Derivar `tasks.md` con las tareas ejecutables.
3. Implementar (paso 5) únicamente después de los anteriores.
