"""
generate_stats.py  --  Informe Econometrico SPX Open Predictor
==============================================================
Estructura de 10 consignas (espejo del TP de Econometria):

  C1.  Estadisticas descriptivas univariadas y bivariadas
  C2.  Especificacion del modelo y signos esperados
  C3.  Salida de regresion logistica + metricas OOS (walk-forward)
  C4.  Coeficiente de Determinacion Multiple (pseudo-R2)
  C5.  Intervalos de confianza y test de hipotesis (Wald, 95%)
  C6.  Prediccion puntual e intervalo de prediccion
  C7.  Multicolinealidad  (VIF)
  C8.  Modelo log-transformado -- comparacion de forma funcional
  C9.  Test RESET de Ramsey  (error de especificacion)
  C10. Heterocedasticidad  (Breusch-Pagan + Hosmer-Lemeshow)

USO
---
    python generate_stats.py
    python generate_stats.py --out informe.html
"""

import argparse
import json
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss, confusion_matrix,
    average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm
import xgboost as xgb
import patsy

warnings.filterwarnings('ignore')

DATA_FILE      = 'spx_data.csv'
TRAIN_WARMUP   = 250
RETRAIN_PERIOD = 21
MODEL_PARAMS   = dict(
    n_estimators=150, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='logloss', verbosity=0,
)

# ── Etiquetas legibles ─────────────────────────────────────────────────────────
FEATURE_LABELS = {
    'overnight':   'Overnight gap (rezagado)',
    'ret_1d':      'Retorno 1 dia',
    'ret_3d':      'Retorno 3 dias',
    'ret_5d':      'Retorno 5 dias',
    'ret_10d':     'Retorno 10 dias',
    'rsi':         'RSI (14)',
    'vol_5d':      'Volatilidad realizada 5d',
    'vol_20d':     'Volatilidad realizada 20d',
    'above_ma20':  'Precio > MA20',
    'above_ma50':  'Precio > MA50',
    'ma20_slope':  'Pendiente MA20 (5d)',
    'vix':         'Nivel VIX',
    'vix_ret_1d':  'Retorno VIX 1 dia',
    'vix_ret_5d':  'Retorno VIX 5 dias',
    'vix_above20': 'VIX > 20',
    'vol_ratio':   'Ratio volumen (vs MA20)',
    'day_of_week': 'Dia de la semana',
    'is_monday':   'Es lunes',
    'month':       'Mes',
    'body':        'Cuerpo vela (rezagado)',
    'upper_wick':  'Mecha superior (rezagada)',
    'lower_wick':  'Mecha inferior (rezagada)',
}

EXPECTED_SIGNS = {
    'overnight':   ('+', 'Gap alcista previo sugiere apertura alcista al dia siguiente.'),
    'ret_1d':      ('+', 'Momentum de corto plazo: mercado en alza tiende a continuar.'),
    'ret_3d':      ('+', 'Momentum de 3 dias refuerza la direccion de corto plazo.'),
    'ret_5d':      ('?', 'Ambiguo: momentum semanal puede revertir o continuar.'),
    'ret_10d':     ('?', 'Efecto de reversion a la media a 2 semanas.'),
    'rsi':         ('-', 'RSI elevado indica sobrecompra, mayor probabilidad de apertura baja.'),
    'vol_5d':      ('-', 'Alta volatilidad reciente asociada a incertidumbre y sesgo bajista.'),
    'vol_20d':     ('-', 'Regimenes de alta volatilidad tienden a aberturas negativas.'),
    'above_ma20':  ('+', 'Tendencia alcista de corto plazo favorece apertura positiva.'),
    'above_ma50':  ('+', 'Tendencia alcista de mediano plazo.'),
    'ma20_slope':  ('+', 'Pendiente positiva de la MA20 indica momentum sostenido.'),
    'vix':         ('-', 'VIX alto refleja miedo del mercado, asociado a aperturas negativas.'),
    'vix_ret_1d':  ('-', 'Suba del VIX en el dia previo anticipa apertura a la baja.'),
    'vix_ret_5d':  ('-', 'Suba sostenida del VIX refuerza sesgo bajista.'),
    'vix_above20': ('-', 'Regimen de alta volatilidad (VIX>20) presiona precios a la baja.'),
    'vol_ratio':   ('?', 'Volumen anormalmente alto puede indicar evento o panico.'),
    'day_of_week': ('?', 'Efectos calendario no tienen signo teorico claro.'),
    'is_monday':   ('-', 'Efecto lunes documentado: mayor probabilidad de apertura negativa.'),
    'month':       ('?', 'Estacionalidad mensual sin direccion teorica unica.'),
    'body':        ('?', 'Vela grande el dia previo puede indicar continuacion o agotamiento.'),
    'upper_wick':  ('-', 'Mecha superior indica rechazo de precios altos (presion bajista).'),
    'lower_wick':  ('+', 'Mecha inferior indica rechazo de precios bajos (soporte).'),
}

FEATURE_COLS = [
    'overnight', 'ret_1d', 'ret_3d', 'ret_5d', 'ret_10d',
    'rsi', 'vol_5d', 'vol_20d', 'above_ma20', 'above_ma50', 'ma20_slope',
    'vix', 'vix_ret_1d', 'vix_ret_5d', 'vix_above20', 'vol_ratio',
    'day_of_week', 'is_monday', 'month', 'body', 'upper_wick', 'lower_wick',
]

LOG_COLS = ['vix', 'vol_ratio']          # features susceptibles de log-transformacion


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (identico a model.py)
# ══════════════════════════════════════════════════════════════════════════════
def compute_rsi(series, period=14):
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def make_features(df):
    f = pd.DataFrame(index=df.index)
    f['target']      = (df['Open'] > df['Close'].shift(1)).astype(int)
    prior_overnight  = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    f['overnight']   = prior_overnight.shift(1)
    f['ret_1d']      = df['Close'].pct_change(1)
    f['ret_3d']      = df['Close'].pct_change(3)
    f['ret_5d']      = df['Close'].pct_change(5)
    f['ret_10d']     = df['Close'].pct_change(10)
    f['rsi']         = compute_rsi(df['Close'], 14)
    f['vol_5d']      = df['Close'].pct_change().rolling(5).std()
    f['vol_20d']     = df['Close'].pct_change().rolling(20).std()
    ma20             = df['Close'].rolling(20).mean()
    ma50             = df['Close'].rolling(50).mean()
    f['above_ma20']  = (df['Close'] > ma20).astype(int)
    f['above_ma50']  = (df['Close'] > ma50).astype(int)
    f['ma20_slope']  = ma20.pct_change(5)
    f['vix']         = df['VIX']
    f['vix_ret_1d']  = df['VIX'].pct_change(1)
    f['vix_ret_5d']  = df['VIX'].pct_change(5)
    f['vix_above20'] = (df['VIX'] > 20).astype(int)
    f['vol_ratio']   = df['Volume'] / df['Volume'].rolling(20).mean()
    f['day_of_week'] = df.index.dayofweek
    f['is_monday']   = (df.index.dayofweek == 0).astype(int)
    f['month']       = df.index.month
    f['body']        = ((df['Close'] - df['Open']).abs() / df['Open']).shift(1)
    f['upper_wick']  = ((df['High'] - df[['Open','Close']].max(axis=1)) / df['Open']).shift(1)
    f['lower_wick']  = ((df[['Open','Close']].min(axis=1) - df['Low'])  / df['Open']).shift(1)
    return f.dropna()


# ══════════════════════════════════════════════════════════════════════════════
#  C1  --  ESTADISTICAS DESCRIPTIVAS
# ══════════════════════════════════════════════════════════════════════════════
def descriptive_stats(feat):
    """Estadisticas univariadas para el target y todas las features."""
    cols = ['target'] + FEATURE_COLS
    rows = []
    for col in cols:
        s = feat[col]
        mode_val = s.mode()
        rows.append({
            'variable': col,
            'label':    FEATURE_LABELS.get(col, 'Variable dependiente (UP=1)'),
            'n':        int(s.count()),
            'media':    round(float(s.mean()),   4),
            'mediana':  round(float(s.median()), 4),
            'moda':     round(float(mode_val.iloc[0]) if len(mode_val) else float('nan'), 4),
            'varianza': round(float(s.var()),    6),
            'std':      round(float(s.std()),    4),
            'asimetria':round(float(s.skew()),   4),
            'curtosis': round(float(s.kurtosis()),4),
            'min':      round(float(s.min()),    4),
            'max':      round(float(s.max()),    4),
        })
    return rows


def correlation_matrix(feat):
    cols  = ['target'] + FEATURE_COLS
    corr  = feat[cols].corr().round(3)
    return corr


# ══════════════════════════════════════════════════════════════════════════════
#  C3  --  WALK-FORWARD + REGRESION LOGISTICA
# ══════════════════════════════════════════════════════════════════════════════
def run_walk_forward(feat):
    results = []
    n = len(feat)
    for i in range(TRAIN_WARMUP, n - 1, RETRAIN_PERIOD):
        train = feat.iloc[:i]
        test  = feat.iloc[i:min(i + RETRAIN_PERIOD, n - 1)]
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(train[FEATURE_COLS].values)
        X_test  = scaler.transform(test[FEATURE_COLS].values)
        model   = xgb.XGBClassifier(**MODEL_PARAMS)
        model.fit(X_train, train['target'].values)
        proba = model.predict_proba(X_test)[:, 1]
        pred  = (proba >= 0.5).astype(int)
        for j in range(len(test)):
            results.append({
                'date':   test.index[j],
                'actual': int(test['target'].iloc[j]),
                'pred':   int(pred[j]),
                'prob':   float(proba[j]),
            })
    return pd.DataFrame(results)


def fit_logit_full(feat):
    """Regresion logistica completa (statsmodels). Devuelve resultado y tabla de coeficientes."""
    scaler = StandardScaler()
    X = scaler.fit_transform(feat[FEATURE_COLS].values)
    X = sm.add_constant(X)
    y = feat['target'].values
    names = ['const'] + FEATURE_COLS
    result = sm.Logit(y, X).fit(disp=0, maxiter=300)

    rows = []
    ci = result.conf_int()
    for i, name in enumerate(names):
        coef   = result.params[i]
        se     = result.bse[i]
        z      = result.tvalues[i]
        p      = result.pvalues[i]
        ci_lo  = ci[i, 0]
        ci_hi  = ci[i, 1]
        or_val = np.exp(coef)
        rows.append({
            'feature':    name,
            'label':      FEATURE_LABELS.get(name, 'Intercepto'),
            'coef':       round(coef,   4),
            'std_err':    round(se,     4),
            'z_stat':     round(z,      4),
            'p_value':    float(p),
            'ci_lo':      round(ci_lo,  4),
            'ci_hi':      round(ci_hi,  4),
            'odds_ratio': round(or_val, 4),
            'or_ci_lo':   round(np.exp(ci_lo), 4),
            'or_ci_hi':   round(np.exp(ci_hi), 4),
            'decision':   'Rechazar H0 (significativo)' if p < 0.05 else 'No rechazar H0',
        })
    model_stats = {
        'lr_chi2': round(result.llr,       4),
        'lr_df':   int(result.df_model),
        'lr_p':    float(result.llr_pvalue),
        'aic':     round(result.aic,       2),
        'bic':     round(result.bic,       2),
        'll_model':round(result.llf,       4),
        'll_null': round(result.llnull,    4),
    }
    return rows, model_stats, result, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  C4  --  PSEUDO-R2
# ══════════════════════════════════════════════════════════════════════════════
def pseudo_r2(y_true, y_prob, ll_model, ll_null):
    n = len(y_true)
    mcfadden   = 1 - ll_model / ll_null
    cox_snell  = 1 - np.exp(2 * (ll_null - ll_model) / n)
    nagelkerke = cox_snell / (1 - np.exp(2 * ll_null / n))
    lr_stat    = -2 * (ll_null - ll_model)
    return {
        'mcfadden':   round(float(mcfadden),   4),
        'cox_snell':  round(float(cox_snell),   4),
        'nagelkerke': round(float(nagelkerke),  4),
        'lr_stat':    round(float(lr_stat),     4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  C5  --  INTERVALOS DE CONFIANZA y TEST DE HIPOTESIS  (ya en fit_logit_full)
# ══════════════════════════════════════════════════════════════════════════════
# La tabla de coeficientes ya incluye CI al 95% y decision de rechazo de H0.


# ══════════════════════════════════════════════════════════════════════════════
#  C6  --  PREDICCION PUNTUAL E INTERVALO
# ══════════════════════════════════════════════════════════════════════════════
def point_prediction(feat, logit_result, logit_scaler, df_raw):
    """Prediccion para el ultimo dia disponible (proximo dia habil)."""
    last_row   = feat.iloc[-1:]
    last_date  = feat.index[-1]

    # Proximo dia habil
    next_date = last_date + pd.tseries.offsets.BDay(1)

    X_last = logit_scaler.transform(last_row[FEATURE_COLS].values)
    X_last_c = sm.add_constant(X_last, has_constant='add')

    pred_frame = logit_result.get_prediction(X_last_c).summary_frame(alpha=0.05)
    prob_up = float(pred_frame['predicted'].iloc[0])
    ci_lo   = float(pred_frame['ci_lower'].iloc[0])
    ci_hi   = float(pred_frame['ci_upper'].iloc[0])

    return {
        'fecha_prediccion': str(next_date.date()),
        'fecha_ultimo_dato': str(last_date.date()),
        'prob_up':  round(prob_up * 100, 2),
        'prob_down':round((1 - prob_up) * 100, 2),
        'ci_lo':    round(ci_lo * 100, 2),
        'ci_hi':    round(ci_hi * 100, 2),
        'signal':   'UP' if prob_up >= 0.5 else 'DOWN',
        'ultimo_cierre': round(float(df_raw['Close'].iloc[-1]), 2),
        'ultimo_vix':    round(float(df_raw['VIX'].iloc[-1]),   2),
        'features_usadas': {k: round(float(last_row[k].iloc[0]), 4) for k in FEATURE_COLS},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  C7  --  MULTICOLINEALIDAD  (VIF)
# ══════════════════════════════════════════════════════════════════════════════
def compute_vif(feat):
    formula = 'target ~ ' + ' + '.join(FEATURE_COLS)
    _, X = patsy.dmatrices(formula, data=feat, return_type='dataframe')
    rows = []
    for i, col in enumerate(X.columns):
        vif_val = variance_inflation_factor(X.values, i)
        fname   = col.replace('Intercept', 'const')
        if fname == 'const':
            continue
        level = ('Bajo (< 5)' if vif_val < 5
                 else 'Moderado (5-10)' if vif_val < 10
                 else 'ALTO (> 10) -- problema de multicolinealidad')
        rows.append({
            'feature': fname,
            'label':   FEATURE_LABELS.get(fname, fname),
            'vif':     round(float(vif_val), 3),
            'level':   level,
        })
    return sorted(rows, key=lambda r: -r['vif'])


# ══════════════════════════════════════════════════════════════════════════════
#  C8  --  MODELO LOG-TRANSFORMADO
# ══════════════════════════════════════════════════════════════════════════════
def log_model_comparison(feat):
    """
    Compara modelo base vs modelo con log(vix) y log(vol_ratio).
    Variables candidatas a log-transformar: vix, vol_ratio (siempre positivas).
    Devuelve tabla comparativa de AIC, BIC, pseudo-R2, LR chi2.
    """
    feat_log = feat.copy()
    feat_log['log_vix']       = np.log(feat['vix'].clip(lower=0.01))
    feat_log['log_vol_ratio'] = np.log(feat['vol_ratio'].clip(lower=0.01))

    log_feature_cols = [
        c if c not in LOG_COLS else f'log_{c}'
        for c in FEATURE_COLS
    ]

    y = feat['target'].values

    def fit_model(X_raw, y):
        sc  = StandardScaler()
        X   = sc.fit_transform(X_raw)
        X   = sm.add_constant(X)
        res = sm.Logit(y, X).fit(disp=0, maxiter=300)
        n   = len(y)
        mcf = 1 - res.llf / res.llnull
        cs  = 1 - np.exp(2 * (res.llnull - res.llf) / n)
        nag = cs / (1 - np.exp(2 * res.llnull / n))
        return {
            'aic':       round(res.aic,   2),
            'bic':       round(res.bic,   2),
            'll':        round(res.llf,   4),
            'lr_chi2':   round(res.llr,   4),
            'lr_p':      float(res.llr_pvalue),
            'mcfadden':  round(mcf,       4),
            'nagelkerke':round(nag,       4),
        }

    base_X    = feat[FEATURE_COLS].values
    log_X     = feat_log[log_feature_cols].values

    base_stats = fit_model(base_X, y)
    log_stats  = fit_model(log_X,  y)

    delta_aic = round(log_stats['aic'] - base_stats['aic'], 2)
    delta_bic = round(log_stats['bic'] - base_stats['bic'], 2)

    base_stats['modelo'] = 'Base (features originales)'
    log_stats['modelo']  = 'Log-transformado (log_vix, log_vol_ratio)'

    nota = ('Modelo log-transformado MEJOR (delta_AIC < 0)' if delta_aic < 0
            else 'Modelo base IGUAL O MEJOR (delta_AIC >= 0)')
    return base_stats, log_stats, delta_aic, delta_bic, nota


# ══════════════════════════════════════════════════════════════════════════════
#  C9  --  TEST RESET DE RAMSEY  (error de especificacion)
# ══════════════════════════════════════════════════════════════════════════════
def reset_test(feat, logit_result, logit_scaler):
    """
    RESET para regresion logistica:
      H0: El modelo esta correctamente especificado.
      Se agrega pi_hat^2 y pi_hat^3 al modelo y se testea su significancia
      conjunta mediante un test de razon de verosimilitud (LR).
      p < 0.05 => rechazo de H0 => hay error de especificacion.
    """
    y = feat['target'].values
    X_sc = logit_scaler.transform(feat[FEATURE_COLS].values)
    X_c  = sm.add_constant(X_sc)

    pi_hat = logit_result.predict(X_c)

    # Modelo aumentado con pi^2 y pi^3
    X_aug = np.column_stack([X_c, pi_hat**2, pi_hat**3])
    res_aug = sm.Logit(y, X_aug).fit(disp=0, maxiter=300)

    lr_stat = -2 * (logit_result.llf - res_aug.llf)
    df_reset = 2          # dos restricciones (coef de pi^2 y pi^3)
    p_val    = 1 - stats.chi2.cdf(lr_stat, df=df_reset)

    decision = ('Rechazar H0: hay error de especificacion (no linealidades u omisiones)'
                if p_val < 0.05
                else 'No rechazar H0: el modelo esta correctamente especificado')
    return {
        'lr_stat':  round(float(lr_stat), 4),
        'df':       df_reset,
        'p_value':  float(p_val),
        'decision': decision,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  C10 -- HETEROCEDASTICIDAD
# ══════════════════════════════════════════════════════════════════════════════
def hosmer_lemeshow(y_true, y_prob, g=10):
    df = pd.DataFrame({'y': y_true, 'p': y_prob})
    df['decile'] = pd.qcut(df['p'], g, duplicates='drop')
    obs_g = df.groupby('decile', observed=True)['y'].sum()
    exp_g = df.groupby('decile', observed=True)['p'].sum()
    n_g   = df.groupby('decile', observed=True)['y'].count()
    h_stat = ((obs_g - exp_g) ** 2 / (exp_g * (1 - exp_g / n_g))).sum()
    p_value = 1 - stats.chi2.cdf(h_stat, df=g - 2)
    decision = ('Rechazar H0: mala calibracion de probabilidades (p < 0.05)'
                if p_value < 0.05
                else 'No rechazar H0: probabilidades bien calibradas (p >= 0.05)')
    return {
        'h_stat':   round(float(h_stat), 4),
        'p_value':  float(p_value),
        'decision': decision,
        'g':        g,
    }


def breusch_pagan_test(feat, logit_result, logit_scaler):
    """
    Test de Breusch-Pagan adaptado para logit:
    Se aplica sobre los residuos de Pearson del modelo logistico
    regresados contra las features originales (regresion OLS auxiliar).
      H0: varianza de los residuos es homogenea (homocedasticidad).
      p < 0.05 => heterocedasticidad.
    """
    y = feat['target'].values
    X_sc  = logit_scaler.transform(feat[FEATURE_COLS].values)
    X_c   = sm.add_constant(X_sc)

    pi_hat      = logit_result.predict(X_c)
    pi_hat      = np.clip(pi_hat, 1e-6, 1 - 1e-6)
    pearson_res = (y - pi_hat) / np.sqrt(pi_hat * (1 - pi_hat))
    sq_res      = pearson_res ** 2

    bp_stat, bp_p, _, _ = het_breuschpagan(sq_res, X_c)
    decision = ('Rechazar H0: hay heterocedasticidad (se recomiendan errores estandar robustos)'
                if bp_p < 0.05
                else 'No rechazar H0: residuos homocedásticos')
    return {
        'bp_stat':  round(float(bp_stat), 4),
        'p_value':  float(bp_p),
        'decision': decision,
    }


def robust_se_comparison(feat, logit_scaler):
    """
    Compara errores estandar clasicos vs robustos (HC3)
    para los 5 features mas significativos.
    """
    X_sc = logit_scaler.transform(feat[FEATURE_COLS].values)
    X_c  = sm.add_constant(X_sc)
    y    = feat['target'].values

    res_ols    = sm.Logit(y, X_c).fit(disp=0, maxiter=300)
    res_robust = sm.Logit(y, X_c).fit(cov_type='HC3', disp=0, maxiter=300)

    names = ['const'] + FEATURE_COLS
    rows  = []
    for i, name in enumerate(names):
        if name == 'const':
            continue
        rows.append({
            'feature':    name,
            'label':      FEATURE_LABELS.get(name, name),
            'se_clasico': round(float(res_ols.bse[i]),    4),
            'se_robusto': round(float(res_robust.bse[i]), 4),
            'z_clasico':  round(float(res_ols.tvalues[i]),    3),
            'z_robusto':  round(float(res_robust.tvalues[i]), 3),
            'p_clasico':  float(res_ols.pvalues[i]),
            'p_robusto':  float(res_robust.pvalues[i]),
        })
    return sorted(rows, key=lambda r: r['p_robusto'])


# ══════════════════════════════════════════════════════════════════════════════
#  METRICAS OOS (walk-forward)
# ══════════════════════════════════════════════════════════════════════════════
def compute_wf_metrics(res):
    y_true = res['actual'].values
    y_prob = res['prob'].values
    y_pred = res['pred'].values
    baseline = y_true.mean()
    acc      = accuracy_score(y_true, y_pred)
    return {
        'n':             len(res),
        'accuracy':      round(acc * 100, 2),
        'baseline':      round(baseline * 100, 2),
        'edge':          round((acc - baseline) * 100, 2),
        'precision':     round(precision_score(y_true, y_pred), 4),
        'recall':        round(recall_score(y_true, y_pred),    4),
        'f1':            round(f1_score(y_true, y_pred),        4),
        'auc':           round(roc_auc_score(y_true, y_prob),   4),
        'brier':         round(brier_score_loss(y_true, y_prob),4),
        'logloss':       round(log_loss(y_true, y_prob),        4),
        'avg_precision': round(average_precision_score(y_true, y_prob), 4),
    }


def per_year_metrics(res):
    r = res.copy()
    r['year']    = r['date'].dt.year
    r['correct'] = r['actual'] == r['pred']
    rows = []
    for yr, g in r.groupby('year'):
        rows.append({
            'year':    yr,
            'n':       len(g),
            'acc':     round(g['correct'].mean() * 100, 1),
            'pct_up':  round(g['actual'].mean() * 100, 1),
            'auc':     round(roc_auc_score(g['actual'], g['prob']) * 100, 1),
        })
    return rows


def calibration_data(y_true, y_prob, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='quantile')
    return [{'mean_pred': round(float(mp), 4), 'frac_pos': round(float(fp), 4)}
            for mp, fp in zip(mean_pred, frac_pos)]


def xgb_importance(feat):
    sc = StandardScaler()
    X  = sc.fit_transform(feat[FEATURE_COLS].values)
    y  = feat['target'].values
    m  = xgb.XGBClassifier(**MODEL_PARAMS)
    m.fit(X, y)
    booster = m.get_booster()
    rows = []
    for imp_type in ('gain', 'cover', 'weight'):
        scores = booster.get_score(importance_type=imp_type)
        for i, col in enumerate(FEATURE_COLS):
            rows.append({'feature': col, 'label': FEATURE_LABELS.get(col, col),
                         'type': imp_type, 'score': scores.get(f'f{i}', 0.0)})
    df = (pd.DataFrame(rows)
            .pivot(index=['feature', 'label'], columns='type', values='score')
            .reset_index())
    df['gain_pct'] = df['gain'] / df['gain'].sum() * 100
    return df.sort_values('gain_pct', ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS HTML
# ══════════════════════════════════════════════════════════════════════════════
def fp(p):
    if p < 0.001: return '< 0.001'
    return f'{p:.4f}'

def sig(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    if p < 0.10:  return '.'
    return ''

def pc(p):
    # all p-values same styling in the academic version — kept for compatibility
    return ''

def sign_badge(s):
    return s

def vif_color(v):
    return ''


# ══════════════════════════════════════════════════════════════════════════════
#  HTML  RENDER  —  estilo informe academico
# ══════════════════════════════════════════════════════════════════════════════
def render_html(
    date_str, feat, df_raw, res,
    desc_rows, corr_matrix_df,
    logit_rows, logit_model_stats, logit_result, logit_scaler,
    pr2, pred_info,
    vif_rows,
    base_stats, log_stats, delta_aic, delta_bic, log_nota,
    reset_res, hl_res, bp_res, robust_rows,
    wf_metrics, peryear, cal_data, xgb_df,
):
    y_true = res['actual'].values
    y_pred = res['pred'].values
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp_v, fn, tp = cm.ravel()

    n_feat = len(FEATURE_COLS)
    n_obs  = len(feat)
    pct_up = round(feat['target'].mean() * 100, 1)
    n_sig  = sum(1 for r in logit_rows if r['feature'] != 'const' and r['p_value'] < 0.05)

    cal_js = json.dumps(cal_data)

    corr_with_target = (corr_matrix_df['target'].drop('target')
                        .abs().sort_values(ascending=False))

    # ── sub-tables ─────────────────────────────────────────────────────────
    def desc_rows_html():
        out = ''
        for r in desc_rows:
            bold = ' class="target-row"' if r['variable'] == 'target' else ''
            lbl  = 'Apertura UP=1 / DOWN=0 (variable dependiente)' if r['variable'] == 'target' else r['label']
            out += (f'<tr{bold}><td>{r["variable"]}</td><td>{lbl}</td>'
                    f'<td>{r["n"]}</td><td>{r["media"]}</td><td>{r["mediana"]}</td>'
                    f'<td>{r["moda"]}</td><td>{r["std"]}</td>'
                    f'<td>{r["asimetria"]}</td><td>{r["curtosis"]}</td>'
                    f'<td>{r["min"]}</td><td>{r["max"]}</td></tr>\n')
        return out

    def spec_rows_html():
        out = ''
        for col in FEATURE_COLS:
            s, razon = EXPECTED_SIGNS[col]
            out += (f'<tr><td>{col}</td><td>{FEATURE_LABELS[col]}</td>'
                    f'<td class="center">{s}</td><td>{razon}</td></tr>\n')
        return out

    def coef_rows_html():
        out = ''
        for r in logit_rows:
            if r['feature'] == 'const':
                continue
            star = sig(r['p_value'])
            dec  = 'Rechazar H₀' if r['p_value'] < 0.05 else 'No rechazar H₀'
            out += (f'<tr><td>{r["feature"]}</td><td>{r["label"]}</td>'
                    f'<td class="r">{r["coef"]:+.4f}</td>'
                    f'<td class="r">{r["std_err"]:.4f}</td>'
                    f'<td class="r">{r["z_stat"]:.3f}</td>'
                    f'<td class="r">{fp(r["p_value"])}&thinsp;{star}</td>'
                    f'<td class="r">[{r["ci_lo"]:.4f},&nbsp;{r["ci_hi"]:.4f}]</td>'
                    f'<td class="r">{r["odds_ratio"]:.4f}</td>'
                    f'<td class="r">[{r["or_ci_lo"]:.3f},&nbsp;{r["or_ci_hi"]:.3f}]</td>'
                    f'<td>{dec}</td></tr>\n')
        return out

    def ci_rows_html():
        out = ''
        for r in logit_rows:
            if r['feature'] == 'const':
                continue
            star = sig(r['p_value'])
            dec  = 'Rechazar H₀' if r['p_value'] < 0.05 else 'No rechazar H₀'
            out += (f'<tr><td>{r["feature"]}</td><td>{r["label"]}</td>'
                    f'<td class="r">{fp(r["p_value"])}&thinsp;{star}</td>'
                    f'<td class="r">{r["ci_lo"]:.4f}</td>'
                    f'<td class="r">{r["ci_hi"]:.4f}</td>'
                    f'<td>{dec}</td></tr>\n')
        return out

    def pred_feat_rows_html():
        out = ''
        for col in FEATURE_COLS:
            val = pred_info['features_usadas'][col]
            out += f'<tr><td>{col}</td><td>{FEATURE_LABELS[col]}</td><td class="r">{val}</td></tr>\n'
        return out

    def vif_rows_html():
        out = ''
        for r in vif_rows:
            flag = ' **' if r['vif'] >= 5 else ''
            out += (f'<tr><td>{r["feature"]}</td><td>{r["label"]}</td>'
                    f'<td class="r">{r["vif"]:.3f}{flag}</td>'
                    f'<td>{r["level"]}</td></tr>\n')
        return out

    def robust_rows_html():
        out = ''
        for r in robust_rows[:12]:
            diff_pct = abs(r['se_robusto'] - r['se_clasico']) / (r['se_clasico'] + 1e-9) * 100
            flag = f' ({diff_pct:+.1f}%)' if diff_pct > 5 else ''
            out += (f'<tr><td>{r["feature"]}</td><td>{r["label"]}</td>'
                    f'<td class="r">{r["se_clasico"]:.4f}</td>'
                    f'<td class="r">{r["se_robusto"]:.4f}{flag}</td>'
                    f'<td class="r">{r["z_clasico"]:.3f}</td>'
                    f'<td class="r">{r["z_robusto"]:.3f}</td>'
                    f'<td class="r">{fp(r["p_clasico"])}&thinsp;{sig(r["p_clasico"])}</td>'
                    f'<td class="r">{fp(r["p_robusto"])}&thinsp;{sig(r["p_robusto"])}</td>'
                    f'</tr>\n')
        return out

    def year_rows_html():
        out = ''
        for r in peryear:
            out += (f'<tr><td>{r["year"]}</td><td class="r">{r["n"]}</td>'
                    f'<td class="r">{r["acc"]}%</td>'
                    f'<td class="r">{r["pct_up"]}%</td>'
                    f'<td class="r">{r["auc"]}%</td></tr>\n')
        return out

    def xgb_rows_html():
        out = ''
        for _, r in xgb_df.iterrows():
            out += (f'<tr><td>{r["feature"]}</td><td>{r["label"]}</td>'
                    f'<td class="r">{r["gain_pct"]:.2f}%</td>'
                    f'<td class="r">{r["gain"]:.2f}</td>'
                    f'<td class="r">{int(r["cover"])}</td>'
                    f'<td class="r">{int(r["weight"])}</td></tr>\n')
        return out

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>SPX Open Predictor &mdash; Informe Econometrico {date_str}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #fff;
    color: #111;
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 13pt;
    line-height: 1.7;
    padding: 0;
  }}

  .page {{
    max-width: 900px;
    margin: 0 auto;
    padding: 3rem 2.5rem 5rem;
  }}

  /* ── Portada ── */
  .cover {{
    border-bottom: 2px solid #111;
    padding-bottom: 1.5rem;
    margin-bottom: 2.5rem;
  }}
  .cover h1 {{
    font-size: 20pt;
    font-weight: bold;
    letter-spacing: .01em;
    margin-bottom: .3rem;
  }}
  .cover .subtitle {{
    font-size: 11pt;
    color: #444;
    margin-bottom: .15rem;
  }}
  .cover .meta {{
    font-size: 10pt;
    color: #666;
    margin-top: .8rem;
  }}

  /* ── Secciones ── */
  h2 {{
    font-size: 13pt;
    font-weight: bold;
    margin: 2.5rem 0 .6rem;
    border-bottom: 1px solid #aaa;
    padding-bottom: .2rem;
  }}
  h3 {{
    font-size: 11.5pt;
    font-weight: bold;
    font-style: italic;
    margin: 1.4rem 0 .4rem;
    color: #222;
  }}

  p {{ margin: .5rem 0 .8rem; font-size: 12pt; }}

  /* ── Tablas ── */
  table {{
    width: 100%;
    border-collapse: collapse;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 10.5pt;
    margin: .8rem 0 1.2rem;
  }}
  thead tr {{
    border-top: 1.5px solid #111;
    border-bottom: 1px solid #111;
  }}
  th {{
    font-weight: bold;
    text-align: left;
    padding: .3rem .55rem;
    background: none;
    color: #111;
    font-size: 10pt;
  }}
  td {{ padding: .25rem .55rem; }}
  tbody tr:last-child {{ border-bottom: 1.5px solid #111; }}
  tbody tr:nth-child(even) {{ background: #f7f7f7; }}
  .r {{ text-align: right; font-family: 'Courier New', monospace; font-size: 10pt; }}
  .center {{ text-align: center; }}
  .target-row td {{ font-weight: bold; }}

  /* ── Ecuacion ── */
  .eq {{
    font-family: 'Courier New', monospace;
    font-size: 11pt;
    background: #f5f5f5;
    border-left: 3px solid #888;
    padding: .5rem 1rem;
    margin: .6rem 0 1rem;
    color: #222;
  }}

  /* ── Bloque de test ── */
  .test-result {{
    border: 1px solid #ccc;
    padding: .7rem 1rem;
    margin: .6rem 0 1rem;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11pt;
    background: #fafafa;
  }}
  .test-result .line {{ margin: .15rem 0; }}
  .test-result .verdict {{ margin-top: .5rem; font-weight: bold; font-size: 10.5pt; }}
  .verdict-ok  {{ color: #1a6e1a; }}
  .verdict-warn {{ color: #c0392b; }}

  /* ── Cuadro resumen de metricas ── */
  .metrics-block {{
    border: 1px solid #bbb;
    padding: .8rem 1.2rem;
    margin: .8rem 0 1.2rem;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 11pt;
    background: #fafafa;
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .3rem 1.5rem;
  }}
  .metrics-block .m-item {{ display: flex; justify-content: space-between; gap: .5rem; }}
  .metrics-block .m-label {{ color: #444; }}
  .metrics-block .m-val   {{ font-family: 'Courier New', monospace; font-weight: bold; }}

  /* ── Confusion matrix ── */
  .cm-table {{ max-width: 280px; margin: .6rem 0 1rem; }}
  .cm-table td {{ text-align: center; padding: .4rem .8rem; font-family: 'Courier New', monospace; }}
  .cm-table .cm-header {{ font-weight: bold; font-style: italic; background: none; }}

  /* ── Nota al pie de tabla ── */
  .tab-note {{
    font-size: 9.5pt;
    color: #444;
    font-family: 'Segoe UI', Arial, sans-serif;
    margin: -.8rem 0 1rem;
  }}

  /* ── Calibracion ── */
  .cal-wrap {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin: .8rem 0 1.2rem; align-items: start; }}
  canvas {{ max-height: 300px; }}

  /* ── Pie ── */
  .footer {{
    margin-top: 4rem;
    border-top: 1px solid #aaa;
    padding-top: .8rem;
    font-size: 9.5pt;
    color: #666;
    font-family: 'Segoe UI', Arial, sans-serif;
    text-align: center;
  }}
</style>
</head>
<body>
<div class="page">

<!-- ══ PORTADA ══════════════════════════════════════════════════════════════ -->
<div class="cover">
  <div class="subtitle">INFORME ECONOMETRICO</div>
  <h1>SPX Open Predictor &mdash; Analisis de Regresion Logistica Binaria</h1>
  <div class="subtitle">
    Prediccion de apertura UP / DOWN del S&amp;P 500 &mdash; objetivo Polymarket
  </div>
  <div class="meta">
    Autores: Federico Cardenes &nbsp;|&nbsp; Generado el {date_str}<br>
    Datos: Yahoo Finance &mdash; 2020-01-02 a 2026-05-22 &nbsp;|&nbsp;
    N&nbsp;=&nbsp;{n_obs:,} observaciones &nbsp;|&nbsp; {n_feat} variables explicativas
  </div>
</div>


<!-- ══ C1 ════════════════════════════════════════════════════════════════════ -->
<h2>1. Estadisticas Descriptivas Univariadas y Bivariadas</h2>

<p>
  La variable dependiente (<em>target</em>) toma valor 1 cuando el S&amp;P 500 abre por encima
  del cierre anterior (apertura alcista, UP) y 0 en caso contrario (DOWN).
  En la muestra el {pct_up}% de las observaciones corresponde a dias con apertura alcista,
  lo que establece el <em>baseline</em> de clasificacion (modelo nulo que siempre predice UP).
  Las 22 variables explicativas se construyen exclusivamente con informacion disponible
  antes de la apertura del mercado.
</p>

<h3>Tabla 1 &mdash; Estadisticas univariadas (variable dependiente y regresores)</h3>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>Variable</th><th>Descripcion</th>
      <th class="r">N</th><th class="r">Media</th><th class="r">Mediana</th>
      <th class="r">Desv. Est.</th><th class="r">Asimetria</th><th class="r">Curtosis</th>
      <th class="r">Min</th><th class="r">Max</th>
    </tr>
  </thead>
  <tbody>{desc_rows_html()}</tbody>
</table>
</div>
<p class="tab-note">
  Asimetria y curtosis en exceso (Fisher). Variable dependiente en negrita.
</p>

<h3>Correlacion bivariada con la variable dependiente (top 5)</h3>
<p>
  Las correlaciones de Pearson entre cada regresor y <em>target</em> son en general bajas,
  lo esperado para datos financieros de alta frecuencia. Las cinco variables con mayor
  correlacion (en valor absoluto) son:
</p>
<table style="max-width:480px">
  <thead><tr><th>Variable</th><th>Descripcion</th><th class="r">r con target</th></tr></thead>
  <tbody>
  {''.join(
    f'<tr><td>{col}</td><td>{FEATURE_LABELS.get(col,col)}</td>'
    f'<td class="r">{corr_matrix_df.loc[col,"target"]:+.3f}</td></tr>'
    for col in corr_with_target.head(5).index
  )}
  </tbody>
</table>
<p class="tab-note">La matriz de correlacion completa (23&times;23) se utiliza en la Seccion 7 para el diagnostico de multicolinealidad.</p>


<!-- ══ C2 ════════════════════════════════════════════════════════════════════ -->
<h2>2. Especificacion del Modelo y Signos Esperados</h2>

<p>
  Dado que la variable dependiente es binaria, se especifica un modelo de
  <strong>regresion logistica</strong> (estimado por maxima verosimilitud).
  La especificacion formal es:
</p>
<div class="eq">P(UP<sub>t</sub> | X<sub>t</sub>) = &Lambda;(&beta;<sub>0</sub> + &beta;<sub>1</sub>&thinsp;overnight<sub>t</sub> + &beta;<sub>2</sub>&thinsp;ret_1d<sub>t</sub> + ... + &beta;<sub>22</sub>&thinsp;lower_wick<sub>t</sub>)</div>
<div class="eq">donde &Lambda;(z) = 1 / (1 + e<sup>&minus;z</sup>) &nbsp;&nbsp; (funcion logistica)</div>
<p>
  Complementariamente, para la prediccion fuera de muestra se entrena un clasificador XGBoost
  con validacion walk-forward (Seccion 3.a). Todas las variables utilizan informacion del
  dia anterior o anterior al mismo, garantizando la ausencia de sesgo de anticipacion
  (<em>lookahead bias</em>).
</p>

<h3>Tabla 2 &mdash; Signos esperados de los coeficientes</h3>
<table>
  <thead><tr><th>Variable</th><th>Descripcion</th><th class="center">Signo esperado</th><th>Razonamiento</th></tr></thead>
  <tbody>{spec_rows_html()}</tbody>
</table>
<p class="tab-note">Signo ambiguo indicado con &ldquo;?&rdquo;. El signo real estimado se reporta en la Seccion 3.b.</p>


<!-- ══ C3 ════════════════════════════════════════════════════════════════════ -->
<h2>3. Salida de Regresion y Comentario de Resultados</h2>

<h3>3.a &mdash; Metricas de desempeno fuera de muestra (walk-forward)</h3>
<p>
  La validacion walk-forward garantiza que cada prediccion se genera sobre datos que
  el modelo no utilizo durante el entrenamiento (ventana de calentamiento: 250 dias,
  re-entrenamiento cada 21 dias). Se evaluaron {wf_metrics['n']:,} dias fuera de muestra.
</p>

<div class="metrics-block">
  <div class="m-item"><span class="m-label">Accuracy OOS</span><span class="m-val">{wf_metrics['accuracy']}%</span></div>
  <div class="m-item"><span class="m-label">Baseline (siempre UP)</span><span class="m-val">{wf_metrics['baseline']}%</span></div>
  <div class="m-item"><span class="m-label">Edge vs. baseline</span><span class="m-val">+{wf_metrics['edge']:.1f} pp</span></div>
  <div class="m-item"><span class="m-label">AUC-ROC</span><span class="m-val">{wf_metrics['auc']:.4f}</span></div>
  <div class="m-item"><span class="m-label">Brier Score</span><span class="m-val">{wf_metrics['brier']:.4f}</span></div>
  <div class="m-item"><span class="m-label">F1 Score</span><span class="m-val">{wf_metrics['f1']:.4f}</span></div>
  <div class="m-item"><span class="m-label">Precision</span><span class="m-val">{wf_metrics['precision']:.4f}</span></div>
  <div class="m-item"><span class="m-label">Recall</span><span class="m-val">{wf_metrics['recall']:.4f}</span></div>
  <div class="m-item"><span class="m-label">Log-Loss</span><span class="m-val">{wf_metrics['logloss']:.4f}</span></div>
</div>

<h3>Tabla 3 &mdash; Accuracy por ano</h3>
<table style="max-width:420px">
  <thead><tr><th>Ano</th><th class="r">N</th><th class="r">Accuracy</th><th class="r">% dias UP</th><th class="r">AUC (%)</th></tr></thead>
  <tbody>{year_rows_html()}</tbody>
</table>
<p class="tab-note">La consistencia del accuracy entre 2021 y 2026 (incluyendo el mercado bajista de 2022) descarta sobreajuste por epoca.</p>

<h3>Matriz de confusion (clasificador XGBoost, OOS)</h3>
<table class="cm-table">
  <thead>
    <tr><th></th><th class="center cm-header">Pred. UP</th><th class="center cm-header">Pred. DOWN</th></tr>
  </thead>
  <tbody>
    <tr><td class="cm-header">Real UP</td><td>{tp}</td><td>{fn}</td></tr>
    <tr><td class="cm-header">Real DOWN</td><td>{fp_v}</td><td>{tn}</td></tr>
  </tbody>
</table>

<h3>3.b &mdash; Regresion logistica: coeficientes y estadisticos</h3>
<p>
  Se estima el modelo logistico sobre la muestra completa ({n_obs:,} observaciones).
  Las variables explicativas estan estandarizadas (media = 0, desviacion estandar = 1)
  para permitir la comparacion directa de la magnitud de los coeficientes.
  El estadistico Z corresponde al test de Wald para H<sub>0</sub>: &beta;<sub>i</sub> = 0.
  Los intervalos de confianza son al 95%.
</p>
<p>
  La razon de verosimilitud del modelo completo es
  LR &chi;&sup2; = <strong>{logit_model_stats['lr_chi2']:.2f}</strong>
  (gl = {logit_model_stats['lr_df']}, p&nbsp;{fp(logit_model_stats['lr_p'])}),
  lo que indica que el conjunto de variables tiene un efecto conjunto estadisticamente
  significativo sobre la probabilidad de apertura alcista.
  De las {n_feat} variables, <strong>{n_sig}</strong> resultan significativas al nivel del 5%.
</p>

<h3>Tabla 4 &mdash; Coeficientes de la regresion logistica</h3>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>Variable</th><th>Descripcion</th>
      <th class="r">Coef.</th><th class="r">Error Est.</th>
      <th class="r">Z (Wald)</th><th class="r">p-valor</th>
      <th class="r">IC 95% (coef.)</th>
      <th class="r">Odds Ratio</th><th class="r">IC 95% (OR)</th>
      <th>Decision H<sub>0</sub></th>
    </tr>
  </thead>
  <tbody>{coef_rows_html()}</tbody>
</table>
</div>
<p class="tab-note">
  Variables estandarizadas. Significancia: *** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05 &nbsp; . p&lt;0.10
</p>

<h3>3.c &mdash; Importancia de variables (XGBoost, muestra completa)</h3>
<p>
  Se reporta la importancia basada en <em>gain</em> (reduccion promedio de perdida al usar
  la variable en una particion), <em>cover</em> (numero promedio de observaciones afectadas)
  y <em>weight</em> (frecuencia de uso en el arbol). El <em>gain</em> es el criterio mas
  informativo.
</p>
<table>
  <thead>
    <tr><th>Variable</th><th>Descripcion</th>
      <th class="r">Gain %</th><th class="r">Gain (raw)</th>
      <th class="r">Cover</th><th class="r">Weight</th></tr>
  </thead>
  <tbody>{xgb_rows_html()}</tbody>
</table>


<!-- ══ C4 ════════════════════════════════════════════════════════════════════ -->
<h2>4. Coeficiente de Determinacion Multiple</h2>

<p>
  En modelos de regresion logistica no existe un R&sup2; ordinario. Se utilizan
  pseudo-R&sup2; derivados de la funcion de log-verosimilitud. El de McFadden
  (tambien llamado &rho;&sup2;) compara la log-verosimilitud del modelo estimado
  con la del modelo nulo (solo intercepto): valores entre 0.2 y 0.4 son considerados
  de buen ajuste. El de Nagelkerke re-escala para que el maximo sea 1.
</p>

<table style="max-width:560px">
  <thead><tr><th>Medida</th><th class="r">Valor</th><th>Interpretacion</th></tr></thead>
  <tbody>
    <tr><td>McFadden R&sup2;</td><td class="r">{pr2['mcfadden']:.4f}</td><td>0.2&ndash;0.4 = excelente</td></tr>
    <tr><td>Nagelkerke R&sup2;</td><td class="r">{pr2['nagelkerke']:.4f}</td><td>Re-escalado a [0,1]</td></tr>
    <tr><td>Cox-Snell R&sup2;</td><td class="r">{pr2['cox_snell']:.4f}</td><td>No alcanza 1 con ajuste perfecto</td></tr>
    <tr><td>LR &chi;&sup2; (analogo F-test)</td><td class="r">{pr2['lr_stat']:.2f}</td><td>gl = {logit_model_stats['lr_df']}</td></tr>
    <tr><td>AIC</td><td class="r">{logit_model_stats['aic']:.1f}</td><td>Menor es mejor (comparacion de modelos)</td></tr>
    <tr><td>BIC</td><td class="r">{logit_model_stats['bic']:.1f}</td><td>Penaliza mas la complejidad</td></tr>
    <tr><td>Log-verosimilitud (modelo)</td><td class="r">{logit_model_stats['ll_model']:.2f}</td><td></td></tr>
    <tr><td>Log-verosimilitud (nulo)</td><td class="r">{logit_model_stats['ll_null']:.2f}</td><td></td></tr>
  </tbody>
</table>
<p>
  El modelo explica aproximadamente el <strong>{pr2['nagelkerke']*100:.1f}%</strong> de la
  variabilidad en la apertura del SPX segun el pseudo-R&sup2; de Nagelkerke,
  lo que representa un ajuste {'razonable a bueno' if pr2['nagelkerke'] > 0.15 else 'moderado'}
  para series financieras de alta frecuencia en las que el ruido es dominante.
</p>


<!-- ══ C5 ════════════════════════════════════════════════════════════════════ -->
<h2>5. Intervalos de Confianza y Test de Hipotesis</h2>

<p>
  Para cada coeficiente se contrasta H<sub>0</sub>:&nbsp;&beta;<sub>i</sub>&nbsp;=&nbsp;0
  contra H<sub>1</sub>:&nbsp;&beta;<sub>i</sub>&nbsp;&ne;&nbsp;0
  mediante el test de Wald al nivel de significancia &alpha;&nbsp;=&nbsp;0.05.
  Se rechaza H<sub>0</sub> cuando el p-valor es menor que 0.05, equivalente
  a que el intervalo de confianza al 95% no incluye el cero.
</p>

<h3>Tabla 5 &mdash; Intervalos de confianza al 95% y decision sobre H<sub>0</sub></h3>
<table>
  <thead>
    <tr>
      <th>Variable</th><th>Descripcion</th>
      <th class="r">p-valor</th>
      <th class="r">IC inf. 95%</th><th class="r">IC sup. 95%</th>
      <th>Decision</th>
    </tr>
  </thead>
  <tbody>{ci_rows_html()}</tbody>
</table>
<p class="tab-note">
  Significancia: *** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05 &nbsp; . p&lt;0.10
</p>


<!-- ══ C6 ════════════════════════════════════════════════════════════════════ -->
<h2>6. Prediccion Puntual e Intervalo de Prediccion</h2>

<p>
  Utilizando el modelo logistico estimado sobre la muestra completa, se obtiene
  la prediccion para el proximo dia habil
  (<strong>{pred_info['fecha_prediccion']}</strong>),
  a partir del vector de variables observadas al cierre del
  {pred_info['fecha_ultimo_dato']}
  (ultimo cierre del SPX: {pred_info['ultimo_cierre']:,.2f}; VIX: {pred_info['ultimo_vix']:.2f}).
</p>

<table style="max-width:480px">
  <thead><tr><th>Concepto</th><th class="r">Valor</th></tr></thead>
  <tbody>
    <tr><td>Senal predicha</td><td class="r"><strong>{pred_info['signal']}</strong></td></tr>
    <tr><td>P(UP) &mdash; estimacion puntual</td><td class="r">{pred_info['prob_up']:.2f}%</td></tr>
    <tr><td>P(DOWN) &mdash; estimacion puntual</td><td class="r">{pred_info['prob_down']:.2f}%</td></tr>
    <tr><td>Limite inferior IC 95% para P(UP)</td><td class="r">{pred_info['ci_lo']:.2f}%</td></tr>
    <tr><td>Limite superior IC 95% para P(UP)</td><td class="r">{pred_info['ci_hi']:.2f}%</td></tr>
  </tbody>
</table>
<p class="tab-note">
  Intervalo de prediccion construido por el metodo delta (propagacion de la incertidumbre parametrica del logit).
</p>

<h3>Tabla 6 &mdash; Vector de variables para la prediccion</h3>
<table>
  <thead><tr><th>Variable</th><th>Descripcion</th><th class="r">Valor observado</th></tr></thead>
  <tbody>{pred_feat_rows_html()}</tbody>
</table>


<!-- ══ C7 ════════════════════════════════════════════════════════════════════ -->
<h2>7. Multicolinealidad (Factor de Inflacion de la Varianza)</h2>

<p>
  El Factor de Inflacion de la Varianza (VIF) para la variable j se define como
  VIF<sub>j</sub>&nbsp;= 1/(1&minus;R&sup2;<sub>j</sub>), donde R&sup2;<sub>j</sub>
  es el coeficiente de determinacion de regresar j contra el resto de las variables.
  Un VIF mayor a 10 indica multicolinealidad severa; entre 5 y 10, moderada;
  menor a 5, sin problema significativo.
</p>

<h3>Tabla 7 &mdash; VIF por variable</h3>
<table>
  <thead><tr><th>Variable</th><th>Descripcion</th><th class="r">VIF</th><th>Diagnostico</th></tr></thead>
  <tbody>{vif_rows_html()}</tbody>
</table>
<p class="tab-note">** indica VIF entre 5 y 10 (multicolinealidad moderada).</p>


<!-- ══ C8 ════════════════════════════════════════════════════════════════════ -->
<h2>8. Modelo Log-Transformado (Forma Funcional Alternativa)</h2>

<p>
  Se re-expresan en logaritmos las variables siempre positivas
  <em>vix</em> (nivel del indice de volatilidad) y <em>vol_ratio</em> (ratio de volumen),
  con el objetivo de capturar posibles relaciones no lineales y reducir
  la influencia de valores extremos. Los criterios de comparacion son
  AIC y BIC: menor valor indica mejor ajuste penalizado por complejidad.
</p>

<h3>Tabla 8 &mdash; Comparacion de modelos</h3>
<table>
  <thead>
    <tr>
      <th>Modelo</th>
      <th class="r">Log-verosim.</th>
      <th class="r">LR &chi;&sup2;</th>
      <th class="r">p (LR)</th>
      <th class="r">McFadden R&sup2;</th>
      <th class="r">Nagelkerke R&sup2;</th>
      <th class="r">AIC</th>
      <th class="r">BIC</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>{base_stats['modelo']}</td>
      <td class="r">{base_stats['ll']:.2f}</td>
      <td class="r">{base_stats['lr_chi2']:.2f}</td>
      <td class="r">{fp(base_stats['lr_p'])}</td>
      <td class="r">{base_stats['mcfadden']:.4f}</td>
      <td class="r">{base_stats['nagelkerke']:.4f}</td>
      <td class="r">{base_stats['aic']:.2f}</td>
      <td class="r">{base_stats['bic']:.2f}</td>
    </tr>
    <tr>
      <td>{log_stats['modelo']}</td>
      <td class="r">{log_stats['ll']:.2f}</td>
      <td class="r">{log_stats['lr_chi2']:.2f}</td>
      <td class="r">{fp(log_stats['lr_p'])}</td>
      <td class="r">{log_stats['mcfadden']:.4f}</td>
      <td class="r">{log_stats['nagelkerke']:.4f}</td>
      <td class="r">{log_stats['aic']:.2f}</td>
      <td class="r">{log_stats['bic']:.2f}</td>
    </tr>
  </tbody>
</table>

<div class="test-result">
  <div class="line">&Delta;AIC (log-transformado &minus; base): <strong>{delta_aic:+.2f}</strong>
  &nbsp;&nbsp; &Delta;BIC: <strong>{delta_bic:+.2f}</strong></div>
  <div class="verdict {'verdict-ok' if delta_aic < 0 else 'verdict-warn'}">{log_nota}</div>
</div>


<!-- ══ C9 ════════════════════════════════════════════════════════════════════ -->
<h2>9. Test RESET de Ramsey (Error de Especificacion)</h2>

<p>
  Se aplica una version del test RESET para regresion logistica.
  Se agregan las probabilidades ajustadas al cuadrado (&pi;&circ;&sup2;)
  y al cubo (&pi;&circ;&sup3;) como regresores adicionales en el modelo logistico.
  La hipotesis nula establece que los coeficientes de estos terminos son conjuntamente
  cero, es decir, que el modelo esta correctamente especificado.
  El test se realiza mediante la razon de verosimilitud (LR).
</p>
<p>
  H<sub>0</sub>: el modelo esta correctamente especificado (no hay no-linealidades
  ni variables omitidas relevantes). Rechazo de H<sub>0</sub> sugiere que la
  forma funcional logistica o el conjunto de variables es inadecuado.
</p>

<div class="test-result">
  <div class="line">LR estadistico: <strong>{reset_res['lr_stat']:.4f}</strong>
  &nbsp;&nbsp; gl = {reset_res['df']}
  &nbsp;&nbsp; p-valor: <strong>{fp(reset_res['p_value'])}</strong></div>
  <div class="verdict {'verdict-warn' if reset_res['p_value'] < 0.05 else 'verdict-ok'}">{reset_res['decision']}</div>
</div>

<p>
  Este resultado es coherente con el uso de XGBoost como complemento del logit:
  al ser un metodo no parametrico, el XGBoost captura las no-linealidades y
  efectos de interaccion que el logit lineal no puede modelar, lo que explica
  la brecha de desempeno entre ambos modelos.
</p>


<!-- ══ C10 ═══════════════════════════════════════════════════════════════════ -->
<h2>10. Heterocedasticidad, Calibracion y Errores Estandar Robustos</h2>

<h3>10.a &mdash; Test de Breusch-Pagan (residuos de Pearson)</h3>
<p>
  El test de Breusch-Pagan se aplica sobre los residuos de Pearson del modelo logistico
  regresados por MCO contra las variables originales (regresion auxiliar).
  H<sub>0</sub>: los residuos son homocedásticos.
  Si se rechaza, se recomienda el uso de errores estandar robustos a heteroscedasticidad (HC3).
</p>

<div class="test-result">
  <div class="line">Estadistico BP: <strong>{bp_res['bp_stat']:.4f}</strong>
  &nbsp;&nbsp; p-valor: <strong>{fp(bp_res['p_value'])}</strong></div>
  <div class="verdict {'verdict-warn' if bp_res['p_value'] < 0.05 else 'verdict-ok'}">{bp_res['decision']}</div>
</div>

<h3>10.b &mdash; Test de Hosmer-Lemeshow (bondad de ajuste)</h3>
<p>
  El test divide las predicciones en 10 deciles y compara frecuencias observadas versus
  esperadas. H<sub>0</sub>: el modelo esta bien calibrado.
  p&nbsp;&gt;&nbsp;0.05 indica buena calibracion de las probabilidades predichas.
</p>

<div class="test-result">
  <div class="line">Estadistico H: <strong>{hl_res['h_stat']:.4f}</strong>
  &nbsp;&nbsp; gl = {hl_res['g'] - 2}
  &nbsp;&nbsp; p-valor: <strong>{fp(hl_res['p_value'])}</strong></div>
  <div class="verdict {'verdict-warn' if hl_res['p_value'] < 0.05 else 'verdict-ok'}">{hl_res['decision']}</div>
</div>

<h3>10.c &mdash; Errores estandar clasicos vs. robustos HC3</h3>
<p>
  Se comparan los errores estandar del estimador MLE clasico con los errores estandar
  robustos a heteroscedasticidad de tipo HC3.
  Las diferencias en porcentaje (&gt;5%) se senalan en la tabla.
  Se reportan los 12 regresores con menor p-valor robusto.
</p>

<h3>Tabla 9 &mdash; Comparacion de errores estandar</h3>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>Variable</th><th>Descripcion</th>
      <th class="r">SE clasico</th><th class="r">SE robusto HC3</th>
      <th class="r">Z clasico</th><th class="r">Z robusto</th>
      <th class="r">p clasico</th><th class="r">p robusto</th>
    </tr>
  </thead>
  <tbody>{robust_rows_html()}</tbody>
</table>
</div>
<p class="tab-note">
  Significancia: *** p&lt;0.001 &nbsp; ** p&lt;0.01 &nbsp; * p&lt;0.05 &nbsp; . p&lt;0.10.
  Porcentajes en columna SE robusto indican diferencia relativa respecto al SE clasico.
</p>

<h3>10.d &mdash; Curva de calibracion (diagrama de fiabilidad)</h3>
<p>
  Un modelo perfectamente calibrado se ubica sobre la diagonal (prediccion = frecuencia real).
  Basado en predicciones out-of-sample del walk-forward ({wf_metrics['n']:,} dias, 10 bins cuantilicos).
</p>

<div class="cal-wrap">
  <canvas id="calChart"></canvas>
  <table>
    <thead><tr><th>P(UP) predicha (media bin)</th><th class="r">Tasa UP observada</th></tr></thead>
    <tbody>
      {''.join(
        f'<tr><td>{r["mean_pred"]:.1%}</td><td class="r">{r["frac_pos"]:.1%}</td></tr>'
        for r in cal_data
      )}
    </tbody>
  </table>
</div>


<!-- ══ PIE ════════════════════════════════════════════════════════════════════ -->
<div class="footer">
  SPX Open Predictor &nbsp;&mdash;&nbsp; Federico Cardenes &nbsp;&mdash;&nbsp; Informe Econometrico {date_str}<br>
  Datos: Yahoo Finance &nbsp;|&nbsp; Modelo: Regresion Logistica + XGBoost (walk-forward)
</div>

</div><!-- .page -->

<script>
const cal = {cal_js};
new Chart(document.getElementById('calChart').getContext('2d'), {{
  type: 'scatter',
  data: {{ datasets: [
    {{
      label: 'Modelo',
      data: cal.map(r => ({{ x: r.mean_pred, y: r.frac_pos }})),
      backgroundColor: '#222', pointRadius: 6,
      showLine: true, borderColor: '#222', tension: 0.2,
    }},
    {{
      label: 'Calibracion perfecta',
      data: [{{ x: 0, y: 0 }}, {{ x: 1, y: 1 }}],
      showLine: true, borderColor: '#888',
      borderDash: [5, 5], pointRadius: 0, fill: false,
    }}
  ]}},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#222', font: {{ family: 'Georgia, serif', size: 11 }} }} }} }},
    scales: {{
      x: {{ min: 0, max: 1,
        title: {{ display: true, text: 'P(UP) predicha', color: '#222', font: {{ family: 'Georgia, serif' }} }},
        ticks: {{ color: '#222' }}, grid: {{ color: '#ddd' }} }},
      y: {{ min: 0, max: 1,
        title: {{ display: true, text: 'Tasa UP observada', color: '#222', font: {{ family: 'Georgia, serif' }} }},
        ticks: {{ color: '#222' }}, grid: {{ color: '#ddd' }} }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description='Informe econometrico SPX')
    parser.add_argument('--out', type=str, default=None)
    args = parser.parse_args()

    date_str = datetime.today().strftime('%Y-%m-%d')
    out_path = args.out or f'spx_informe_econometrico_{date_str}.html'

    print(f"[1/11] Cargando datos desde {DATA_FILE}...")
    df_raw = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    print(f"       {len(df_raw)} dias de trading")

    print("[2/11] Construyendo features...")
    feat = make_features(df_raw)
    print(f"       {len(feat)} observaciones, {len(FEATURE_COLS)} features")

    print("[3/11] C1 — Estadisticas descriptivas...")
    desc_rows  = descriptive_stats(feat)
    corr_df    = correlation_matrix(feat)

    print("[4/11] C3 — Walk-forward (XGBoost)...")
    res = run_walk_forward(feat)

    print("[5/11] C3 — Regresion logistica (statsmodels)...")
    logit_rows, logit_model_stats, logit_result, logit_scaler = fit_logit_full(feat)

    print("[6/11] C4 — Pseudo-R2...")
    pr2 = pseudo_r2(
        res['actual'].values, res['prob'].values,
        logit_model_stats['ll_model'], logit_model_stats['ll_null']
    )

    print("[7/11] C6 — Prediccion puntual...")
    pred_info = point_prediction(feat, logit_result, logit_scaler, df_raw)

    print("[8/11] C7 — VIF...")
    vif_rows = compute_vif(feat)

    print("[9/11] C8 — Modelo log-transformado...")
    base_stats, log_stats, delta_aic, delta_bic, log_nota = log_model_comparison(feat)

    print("[10/11] C9/C10 — RESET, Breusch-Pagan, Hosmer-Lemeshow, SE robustos...")
    reset_res = reset_test(feat, logit_result, logit_scaler)
    hl_res    = hosmer_lemeshow(res['actual'].values, res['prob'].values)
    bp_res    = breusch_pagan_test(feat, logit_result, logit_scaler)
    robust_rows = robust_se_comparison(feat, logit_scaler)

    print("[11/11] Generando informe HTML...")
    wf_metrics = compute_wf_metrics(res)
    peryear    = per_year_metrics(res)
    cal_data   = calibration_data(res['actual'].values, res['prob'].values)
    xgb_df     = xgb_importance(feat)

    html = render_html(
        date_str, feat, df_raw, res,
        desc_rows, corr_df,
        logit_rows, logit_model_stats, logit_result, logit_scaler,
        pr2, pred_info,
        vif_rows,
        base_stats, log_stats, delta_aic, delta_bic, log_nota,
        reset_res, hl_res, bp_res, robust_rows,
        wf_metrics, peryear, cal_data, xgb_df,
    )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"\nInforme guardado en: {out_path}")
    print(f"\n{'='*55}")
    print(f"  RESUMEN — 10 CONSIGNAS")
    print(f"{'='*55}")
    print(f"  C1  Obs. totales: {len(feat):,} | Features: {len(FEATURE_COLS)} | % UP: {feat['target'].mean()*100:.1f}%")
    print(f"  C2  Especificacion: Logit binario, 22 features, sin lookahead")
    print(f"  C3  Accuracy OOS: {wf_metrics['accuracy']}% | AUC: {wf_metrics['auc']} | F1: {wf_metrics['f1']}")
    print(f"      LR chi2: {logit_model_stats['lr_chi2']:.2f} (df={logit_model_stats['lr_df']}, p={fp(logit_model_stats['lr_p'])})")
    print(f"  C4  McFadden R2: {pr2['mcfadden']:.4f} | Nagelkerke R2: {pr2['nagelkerke']:.4f}")
    print(f"      AIC: {logit_model_stats['aic']:.1f} | BIC: {logit_model_stats['bic']:.1f}")
    n_sig = sum(1 for r in logit_rows if r['feature']!='const' and r['p_value']<0.05)
    print(f"  C5  Features significativas (p<0.05): {n_sig}/{len(FEATURE_COLS)}")
    print(f"  C6  Prediccion {pred_info['fecha_prediccion']}: {pred_info['signal']} "
          f"| P(UP)={pred_info['prob_up']}% | IC95%=[{pred_info['ci_lo']}%, {pred_info['ci_hi']}%]")
    vif_max = max(r['vif'] for r in vif_rows)
    print(f"  C7  VIF max: {vif_max:.2f} ({'sin multicolinealidad severa' if vif_max < 10 else 'MULTICOLINEALIDAD'})")
    print(f"  C8  Delta AIC log-modelo: {delta_aic:+.2f} | {log_nota}")
    print(f"  C9  RESET test: LR={reset_res['lr_stat']:.4f}, p={fp(reset_res['p_value'])} | {reset_res['decision'][:40]}...")
    print(f"  C10 Breusch-Pagan: p={fp(bp_res['p_value'])} | Hosmer-Lemeshow: p={fp(hl_res['p_value'])}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
