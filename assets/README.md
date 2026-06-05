# assets/

Activos estáticos del proyecto: el modelo y el manifiesto del dataset.

```
assets/
├── sam3/              # pesos del modelo SAM3 (reales, git-ignored)
└── db_metadata.csv    # manifiesto de los videos de data/raw (versionado)
```

- **`sam3/`** — checkpoint de SAM3 (`sam3.pt`). No se versiona; se coloca/descarga
  por separado (ver el futuro script `bootstrap_data`).
- **`db_metadata.csv`** — manifiesto del dataset, generado por
  `src.data.build_metadata_csv`. Sí se versiona.

## `db_metadata.csv`

Una fila por video `.MOV` descubierto bajo `data/raw/`. Permite trabajar el dataset
(splits, análisis) sin abrir cada video. Se regenera con
`build_metadata_csv(force=True)`.

| Columna       | Tipo  | Descripción                                              |
|---------------|-------|----------------------------------------------------------|
| `id`          | int   | Índice secuencial `0..N-1` (orden alfabético de `ruta`). |
| `ruta`        | str   | Ruta relativa a la raíz del proyecto (POSIX).            |
| `nombre`      | str   | Nombre del archivo con extensión.                        |
| `duracion`    | float | Duración en segundos.                                    |
| `ancho`       | int   | Resolución horizontal (px).                              |
| `alto`        | int   | Resolución vertical (px).                                |
| `fps_average` | float | Cuadros por segundo promedio.                            |
| `split`       | int   | Partición del dataset: `0`, `1` o `2` (ver abajo).       |

### Columna `split`

Indica a qué partición pertenece cada video. El reparto es **aleatorio
reproducible** (semilla en `seeds.split` del config); los videos listados en
`splits.forced_testing` quedan siempre en testing.

| Valor | Partición   | Uso                                       | Cantidad |
|-------|-------------|-------------------------------------------|----------|
| `0`   | Reserva     | No se usa por ahora; resto del dataset.   | ~80      |
| `1`   | Fine-tuning | Entrenamiento/ajuste del detector.        | 23       |
| `2`   | Testing     | Evaluación del pipeline.                   | 20       |

Los splits son **disjuntos** (cada video en exactamente uno) y cubren todos los
videos. Cambiar `seeds.split` o `splits.forced_testing` y regenerar redistribuye las
particiones manteniendo esos conteos.
