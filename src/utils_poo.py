from typing import Union, Dict, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

logger = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, pd.Series]


class CreditScoreAnalysisError(Exception):
    """
    [pt] Exceção base para erros de análise de score de crédito.
    [en] Base exception for credit score analysis errors.
    """
    pass


class CreditScoreAnalysis:
    """
    Credit score analysis for bad rate and related statistics by percentile.

    Parameters
    ----------
    scores : array-like
        Model scores (higher = riskier).
    target : array-like
        Binary target (1 = bad, 0 = good).
    n_percentiles : int, default=10
        Number of equal-frequency bins.
    model_name : str, optional
        Label added as a column in the output DataFrame.

    Raises
    ------
    CreditScoreAnalysisError
        If inputs fail validation.
    """

    def __init__(
            self,
            scores: ArrayLike,
            target: ArrayLike,
            n_percentiles: int = 10,
            model_name: Optional[str] = None
            ) -> None:

        self.scores = scores
        self.target = target
        self.n_percentiles = n_percentiles
        self.model_name = model_name
        self._validate()

    def _validate(self) -> None:
        """
        [pt] Valida scores, target e parâmetros antes do cálculo.
        [en] Validate scores, target, and parameters before computation.
        """
        if len(self.scores) != len(self.target):
            raise CreditScoreAnalysisError(
                f"[pt] scores e target têm tamanhos diferentes: "
                f"{len(self.scores)} vs {len(self.target)}.\n"
                f"[en] scores and target have different lengths: "
                f"{len(self.scores)} vs {len(self.target)}."
            )

        if len(self.scores) == 0:
            raise CreditScoreAnalysisError(
                "[pt] scores e target não podem ser vazios.\n"
                "[en] scores and target must not be empty."
            )

        target_values = set(np.unique(self.target))
        if not target_values.issubset({0, 1}):
            raise CreditScoreAnalysisError(
                f"[pt] target deve conter apenas 0 e 1. Valores encontrados: {target_values}\n"
                f"[en] target must contain only 0 and 1. Found values: {target_values}."
            )

        if np.sum(self.target) == 0:
            raise CreditScoreAnalysisError(
                "[pt] target não possui nenhum mau (classe 1).\n"
                "[en] target has no bad observations (class 1)."
            )

        if pd.isna(self.scores).any():
            n_null = int(pd.isna(self.scores).sum())
            logger.warning(
                "[pt] %d score(s) nulos encontrados e serão ignorados no cálculo. "
                "[en] %d null score(s) found and will be ignored during computation.",
                n_null, n_null
            )

        if not isinstance(self.n_percentiles, int) or self.n_percentiles < 2:
            raise CreditScoreAnalysisError(
                f"[pt] n_percentiles deve ser um inteiro >= 2. Recebido: {self.n_percentiles}\n"
                f"[en] n_percentiles must be an integer >= 2. Received: {self.n_percentiles}."
            )

        if self.n_percentiles > len(self.scores):
            raise CreditScoreAnalysisError(
                f"[pt] n_percentiles ({self.n_percentiles}) maior que o número de observações "
                f"({len(self.scores)}).\n"
                f"[en] n_percentiles ({self.n_percentiles}) cannot be greater than "
                f"the number of observations ({len(self.scores)})."
            )

    def _assign_percentiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [pt] Atribui faixas de percentil ao DataFrame.
        [en] Assign equal-frequency percentile bins to the score column.
        """
        df["percentile"] = pd.qcut(
            df["score"],
            q=self.n_percentiles,
            labels=False,
            duplicates="drop",
        )

        df["percentile"] = df["percentile"] + 1

        df["percentile_rev"] = self.n_percentiles - df["percentile"] + 1

        return df

    def _aggregate_by_percentile(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        [pt] Agrega total, maus e bons por percentil.
        [en] Aggregate total, bads, and goods per percentile bin.
        """
        result = (
            df.groupby("percentile", as_index=False)
            .agg(
                total=("target", "count"),
                bads=("target", "sum"),
                goods=("target", lambda x: (x == 0).sum()),
            )
            .sort_values("percentile")
            .reset_index(drop=True)
        )

        result["percentile_rev"] = self.n_percentiles - result["percentile"] + 1
        return result

    def _add_cumulative_metrics(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        [pt] Adiciona métricas cumulativas (sentido crescente de risco).
        [en] Add forward cumulative metrics (ascending risk order).
        """
        result["total_cum"] = result["total"].cumsum()
        result["bads_cum"] = result["bads"].cumsum()
        result["goods_cum"] = result["goods"].cumsum()

        total = result["total"].sum()
        total_bads = result["bads"].sum()

        result["total_rate_cum"] = result["total_cum"] / total
        result["gain"] = result["bads_cum"] / total_bads
        result["lift"] = result["gain"] / result["total_rate_cum"]
        result["bad_rate_cum"] = result["bads_cum"] / result["total_cum"]
        result["good_rate_cum"] = result["goods_cum"] / result["total_cum"]
        return result

    def _add_reverse_cumulative_metrics(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        [pt] Adiciona métricas cumulativas reversas (do pior score para o melhor).
        [en] Add reverse cumulative metrics (from worst score to best).
        """
        total = result["total"].sum()
        total_bads = result["bads"].sum()

        result["bads_cum_rev"] = result["bads"][::-1].cumsum()[::-1]
        result["total_cum_rev"] = result["total"][::-1].cumsum()[::-1]
        result["bad_rate_cum_rev"] = result["bads_cum_rev"] / result["total_cum_rev"]
        result["lift_rev"] = (result["bads_cum_rev"] / total_bads) / (result["total_cum_rev"] / total)
        result["gain_rev"] = result["bads_cum_rev"] / total_bads

        return result

    def _rename_percentiles(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        [pt] Adiciona colunas de label para percentile e percentile_rev.
        [en] Add display label columns for percentile and percentile_rev.
        """

        result["percentile_label"] = result["percentile"].astype(str)
        result.loc[result["percentile"] == 1, "percentile_label"] = f"1 - Menos arriscado"
        result.loc[result["percentile"] == self.n_percentiles, "percentile_label"] = f"{self.n_percentiles} - Mais arriscado"

        result["percentile_rev_label"] = result["percentile_rev"].astype(str)
        result.loc[result["percentile_rev"] == 1, "percentile_rev_label"] = f"1 - Mais arriscado"
        result.loc[result["percentile_rev"] == self.n_percentiles, "percentile_rev_label"] = f"{self.n_percentiles} - Menos arriscado"

        return result

    def compute_percentiles_metrics(self) -> pd.DataFrame:
        """
        [pt] Orquestra o cálculo de todas as métricas por percentil.
        [en] Compute bad rate and ordering statistics by percentile.
        """
        df = pd.DataFrame({"score": self.scores, "target": self.target})
        df = self._assign_percentiles(df)

        result = self._aggregate_by_percentile(df)
        result = self._add_cumulative_metrics(result)
        result = self._add_reverse_cumulative_metrics(result)

        result["bad_rate"]   = result["bads"] / result["total"]

        result = self._rename_percentiles(result)

        if self.model_name is not None:
            result["model"] = self.model_name

        logger.info(
            "Métricas calculadas para '%s': %d percentis, %d observações.",
            self.model_name or "unnamed",
            result["percentile"].nunique(),
            result["total"].sum(),
        )
        return result



def analyze_multiple_models(
    models: Dict[str, ArrayLike],
    target: ArrayLike,
    n_percentiles: int = 10,
) -> pd.DataFrame:
    """
    [pt] Executa análise percentil para múltiplos modelos de crédito.
    [en] Run percentile-based bad rate analysis for multiple credit models.

    Parameters
    ----------
    models : dict[str, array-like]
        Dictionary mapping model names to score arrays.
    target : array-like
        Binary target (1 = bad, 0 = good).
    n_percentiles : int, default=10
        Number of equal-frequency bins.

    Returns
    -------
    pd.DataFrame
        Concatenated analysis results for all models,
        with a 'model' column identifying each one.

    Raises
    ------
    ValueError
        If models dict is empty.
    """
    if not models:
        raise ValueError("[pt] O dicionário 'models' não pode ser vazio.\n"
                         "[en] 'models' dictionary must not be empty.")

    results = [
        CreditScoreAnalysis(
            scores=scores,
            target=target,
            n_percentiles=n_percentiles,
            model_name=name,
        ).compute_percentiles_metrics()
        for name, scores in models.items()
    ]

    return pd.concat(results, ignore_index=True)


def compute_roc_curves(
    models: Dict[str, ArrayLike],
    target: ArrayLike,
) -> Dict[str, pd.DataFrame]:
    """
    [pt] Calcula os dados da curva ROC para múltiplos modelos.
    [en] Compute ROC curve data for multiple credit models.

    Parameters
    ----------
    models : dict[str, array-like]
        Dictionary mapping model names to score arrays (probability of bad).
    target : array-like
        Binary target (1 = bad, 0 = good).

    Returns
    -------
    dict[str, pd.DataFrame]
        Dict mapping model name → DataFrame with columns:
        fpr, tpr, thresholds, auc, gini
    """
    result = {}
    for name, scores in models.items():
        fpr, tpr, thresholds = roc_curve(target, scores)
        auc  = roc_auc_score(target, scores)
        gini = 2 * auc - 1
        result[name] = pd.DataFrame({
            "fpr":        fpr,
            "tpr":        tpr,
            "thresholds": thresholds,
            "auc":        auc,
            "gini":       gini,
        })
    return result


def compute_ks_curves(
    models: Dict[str, ArrayLike],
    target: ArrayLike,
    n_bins: int = 100,
) -> Dict[str, pd.DataFrame]:
    """
    [pt] Calcula os dados da curva KS (distribuições acumuladas de bons e maus).
    [en] Compute KS curve data (cumulative bad/good distributions) for multiple models.

    Uses continuous binning (pd.cut) rather than percentile discretization,
    producing a smoother and more precise KS curve — standard in credit modeling.

    Parameters
    ----------
    models : dict[str, array-like]
        Dictionary mapping model names to score arrays (probability of bad).
    target : array-like
        Binary target (1 = bad, 0 = good).
    n_bins : int, default=100
        Number of bins for score discretization.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dict mapping model name → DataFrame with columns:
        score, cum_bads_pct, cum_goods_pct, ks_spread
    """
    result = {}
    for name, scores in models.items():
        df = pd.DataFrame({"score": np.array(scores), "target": np.array(target)})
        df["bucket"] = pd.cut(df["score"], bins=n_bins)

        ks_table = (
            df.groupby("bucket", observed=False)
              .agg(max_score=("score", "max"),
                   bads=("target", "sum"),
                   total=("target", "count"))
              .reset_index()
              .sort_values("max_score")
        )
        ks_table["goods"]         = ks_table["total"] - ks_table["bads"]
        ks_table["cum_bads_pct"]  = ks_table["bads"].cumsum()  / ks_table["bads"].sum()
        ks_table["cum_goods_pct"] = ks_table["goods"].cumsum() / ks_table["goods"].sum()
        ks_table["ks_spread"]     = np.abs(ks_table["cum_bads_pct"] - ks_table["cum_goods_pct"])

        result[name] = ks_table.rename(columns={"max_score": "score"})[
            ["score", "cum_bads_pct", "cum_goods_pct", "ks_spread"]
        ]
    return result


print("utils_poo carregado")