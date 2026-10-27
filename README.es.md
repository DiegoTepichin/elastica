# elastica

Estima qué tan sensibles al precio son realmente tus clientes a partir de datos de ventas comunes, con double machine learning, y conviértelo en una recomendación de precio que puedas defender.

[![CI](https://github.com/DiegoTepichin/elastica/actions/workflows/ci.yml/badge.svg)](https://github.com/DiegoTepichin/elastica/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

[Read in English](README.md)

![Elasticidad estimada contra la real en 50 productos simulados: el OLS ingenuo y el OLS con controles lineales están sesgados hacia cero; elastica cae sobre la diagonal](https://raw.githubusercontent.com/DiegoTepichin/elastica/main/docs/benchmark.svg)

## Por qué

La forma habitual de medir la elasticidad precio, una regresión de log cantidad contra log precio, falla siempre que algo mueve precio y demanda al mismo tiempo. Los comercios suben precios en temporada alta, cuando la demanda ya es alta, así que la pendiente ingenua dice que los clientes casi no reaccionan al precio. En el benchmark de arriba no solo se equivoca: para muchos productos sale **positiva**.

`elastica` elimina esa confusión con double machine learning (Chernozhukov et al., 2018): predice precio y demanda a partir del calendario y de tus controles con gradient boosting, y estima la elasticidad con lo que queda. Obtienes un intervalo de confianza, diagnósticos que te dicen cuándo no confiar en el número y una recomendación de precio que no se aleja de los precios que de verdad has probado.

## Benchmark

`examples/benchmark.py` simula 50 productos con elasticidad conocida, estacionalidad y dos picos de ventas de una semana que suben precio y demanda, y compara tres estimadores sobre los mismos datos:

| Método | Error absoluto medio | Sesgo medio |
|---|---|---|
| OLS ingenuo (log q contra log p) | 1.59 | +1.59 |
| OLS + los mismos controles, lineales | 0.55 | +0.55 |
| **elastica (DML)** | **0.12** | **+0.09** |

El intervalo de confianza al 95 % de elastica contiene el valor real en el 80 % de las corridas. La diferencia contra 95 % viene de los picos de ventas: con dos años de datos diarios los modelos auxiliares no los aprenden del todo, lo que deja un sesgo pequeño hacia cero. Sin los picos, la cobertura es de 19 de 20 corridas. Reprodúcelo con `uv run --group docs python examples/benchmark.py`.

## Instalación

```bash
pip install elastica
```

Requiere Python 3.11 o superior. Dependencias: numpy, pandas, scikit-learn.

## Inicio rápido

Genera datos de ejemplo con respuestas conocidas y después estima y calcula precios (desde un clon de este repositorio, antepón `uv run` a cada comando):

```text
$ elastica simulate -o sales.csv
true elasticities: SKU-001 -2.2, SKU-002 -1.87, SKU-003 -1.53, SKU-004 -1.2

$ elastica estimate sales.csv --controls promo
Product  Elasticity  95% CI low  95% CI high  Naive  Periods
-------  ----------  ----------  -----------  -----  -------
SKU-001       -2.02       -2.23        -1.81  -0.55      730
SKU-002       -1.77       -1.97        -1.58  -0.32      730
SKU-003       -1.47       -1.66        -1.27  -0.05      730
SKU-004       -1.07       -1.26        -0.89   0.35      730

$ elastica price sales.csv --controls promo --cost unit_cost
Product  Elasticity  Current  Recommended  Range low  Range high  Profit  Revenue
-------  ----------  -------  -----------  ---------  ----------  ------  -------
SKU-001       -2.02   105.34       109.15      99.95      123.10   +0.1%    -3.6%
SKU-002       -1.77    67.42        72.35      64.06       80.91   +0.4%    -5.3%
SKU-003       -1.47   128.62       154.35     146.68      154.35   +4.6%    -8.1%
SKU-004       -1.07    96.10       115.33     115.33      115.33  +13.1%    -1.3%
```

*Range* es el precio recomendado en cada extremo del intervalo de confianza de la elasticidad. *Profit* y *Revenue* son los cambios esperados por periodo en utilidad e ingreso si se cumplen los supuestos del modelo.

## Tus datos

Una fila por venta o por producto y día, en CSV o Parquet. Solo tres columnas son obligatorias y pueden llamarse como quieras:

| Campo | Opción | Obligatorio | Notas |
|---|---|---|---|
| Fecha | `--date` | sí | Cualquier formato que pandas entienda. |
| Precio | `--price` | sí | Precio unitario pagado. |
| Cantidad | `--quantity` | sí | Unidades vendidas. |
| Producto | `--product` | no | Usa `--product ''` si el archivo tiene un solo producto. |
| Costo unitario | `--cost` | no | Necesario para maximizar utilidad; o `--unit-cost 42` para todos los productos. |
| Controles | `--controls a,b` | no | Todo lo demás que mueva precio o demanda: promociones, tienda, canal, quiebres de inventario. Las columnas de texto se codifican one-hot. |

Las filas se agregan a producto × día (`--frequency W` o `M` para semanal o mensual) con un precio ponderado por cantidad. Las variables de calendario (tendencia, mes, día del año, semana del año, día de la semana) se agregan solas. Los periodos sin ventas se descartan y se cuentan.

## Cómo funciona

Para cada producto, en logaritmos:

```
log q = θ · log p + g(X) + ε      θ es la elasticidad
log p =         m(X) + v
```

1. **Cross-fitting.** `g` y `m` se aprenden con gradient boosting en K−1 particiones y se predicen en la que quedó fuera, así ninguna observación la predice un modelo que la vio.
2. **Partialling out.** `θ` es la pendiente de los residuos de la demanda contra los residuos del precio: la parte de la variación del precio que el calendario y los controles no explican.
3. **Inferencia.** Los errores estándar salen del score ortogonal de Neyman; todo el procedimiento se repite con nuevas particiones (3 por defecto) y se agrega con la mediana.
4. **Precio.** La demanda se aproxima con elasticidad constante alrededor del precio actual, así que el precio que maximiza la utilidad es `costo · θ / (1 + θ)` cuando `θ < −1`. El resultado se limita a una banda alrededor del precio actual (80 %–120 % por defecto), porque la aproximación solo es confiable cerca de los precios observados.

Cada estimación reporta junto al DML la pendiente ingenua y un OLS con controles lineales, más el R² con cross-fitting de ambos modelos auxiliares.

### Advertencias que puedes ver

| Advertencia | Significado |
|---|---|
| weakly identified | Los controles explican casi todos los cambios de precio. Queda muy poca variación independiente para aprender; haz una prueba de precios. |
| positive elasticity | La demanda sube con el precio. Probablemente falta un confusor en `--controls`. |
| confidence interval includes −1 | Los datos no distinguen entre demanda elástica e inelástica; la dirección del cambio óptimo es incierta. |
| capped at the band | El óptimo está fuera de los precios que has probado. Acércate a él con pasos pequeños y medidos. |

## API de Python

```python
import elastica

sim = elastica.simulate()
panel = elastica.prepare(sim.transactions, elastica.Schema(cost="unit_cost", controls=("promo",)))

for item in elastica.estimate(panel):
    r = item.result
    rec = elastica.recommend(item)
    print(
        f"{item.product}: {r.elasticity:.2f} [{r.ci_low:.2f}, {r.ci_high:.2f}]  naive {r.naive_elasticity:+.2f}  ->  {rec.current_price:.2f} to {rec.recommended_price:.2f} ({rec.reason})"
    )
```

Más:

- `elastica.estimate(panel, pooled=True)`: una sola estimación para todos los productos con efectos fijos por producto, para SKUs con poca historia.
- `elastica.EstimateOptions(learner=..., folds=Folds.TIME, n_folds=5, n_repeats=3, confidence=0.9)`: cualquier regresor de scikit-learn sirve como `learner`. Las particiones por bloques de tiempo son más conservadoras y necesitan unas tres temporadas de historia.
- `elastica.estimate_elasticity(y, d, x)`: el estimador sobre arreglos.
- `elastica.demand_curve(...)`, `estimates_frame`, `recommendations_frame`: tablas para graficar y reportar.

## Límites

- Es causal solo bajo el supuesto habitual: todo lo que mueve a la vez precio y demanda está en la fecha o en `--controls`. Si el precio de un competidor mueve el tuyo y no está en los datos, la estimación tiene sesgo.
- Solo elasticidad propia; no modela efectos cruzados ni sustitución entre productos.
- La recomendación de precio es el punto de partida de una prueba, no una orden.

## Desarrollo

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy .
```

Ver [CONTRIBUTING.md](CONTRIBUTING.md).

## Licencia

[MIT](LICENSE) © Diego Tepichin
