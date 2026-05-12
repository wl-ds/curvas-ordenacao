from typing import Optional, Tuple
import warnings

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes

plt.style.use("seaborn-v0_8")

FigAx = Tuple[plt.Figure, matplotlib.axes.Axes]

class PercentilePlots:
    """
    Classe para criação de gráficos de análise por percentil.
    """

    def __init__(
        self,
        figsize: Tuple[int, int] = (5, 4),
        palette: str = "viridis",
        title_fontsize: int = 12,
        label_fontsize: int = 10,
        tick_fontsize: int = 9,
        legend_fontsize: int = 9,
        dpi: int = 300,
    ) -> None:

        self.figsize = figsize
        self.palette = palette
        self.title_fontsize = title_fontsize
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize
        self.legend_fontsize = legend_fontsize
        self.dpi = dpi

    def _save_figure(
        self,
        fig: plt.Figure,
        output_path: Optional[str],
    ) -> None:
        """
        Salva a figura se output_path for informado.
        """

        if output_path is not None:
            fig.savefig(
                output_path,
                dpi=self.dpi,
                bbox_inches="tight",
            )

    def _validate_results_percentil(
        self,
        results_percentil: pd.DataFrame,
        required_cols: list,
    ) -> None:
        """
        Valida se as colunas necessárias existem.
        """

        missing = [
            col
            for col in required_cols
            if col not in results_percentil.columns
        ]

        if missing:
            raise ValueError(
                f"results_percentil está sem as colunas: {missing}"
            )

    def _compute_cutoff(
        self,
        n_percentiles: int,
    ) -> int:
        """
        Calcula o ponto médio dos percentis.
        """

        if n_percentiles % 2 != 0:
            warnings.warn(
                (
                    f"n_percentiles={n_percentiles} é ímpar. "
                    f"O cutoff será {n_percentiles // 2}."
                ),
                UserWarning,
                stacklevel=2,
            )

        return n_percentiles // 2

    def _apply_font_sizes(
        self,
        ax: matplotlib.axes.Axes,
    ) -> None:
        """
        Aplica tamanhos de fonte no gráfico.
        """

        ax.title.set_fontsize(
            self.title_fontsize
        )

        ax.xaxis.label.set_fontsize(
            self.label_fontsize
        )

        ax.yaxis.label.set_fontsize(
            self.label_fontsize
        )

        ax.tick_params(
            axis="both",
            labelsize=self.tick_fontsize,
        )

        if ax.get_legend() is not None:
            ax.legend(
                fontsize=self.legend_fontsize
            )

    def plot_bad_rate(
        self,
        results_percentil: pd.DataFrame,
        target: pd.Series,
        title: Optional[str] = None,
        xlabel: str = "Decil",
        ylabel: str = "Taxa de Mau",
        output_path: Optional[str] = None,
    ) -> FigAx:
        """
        Gráfico de taxa de mau por percentil.
        """

        self._validate_results_percentil(
            results_percentil,
            ["percentile", "bad_rate", "model"],
        )

        avg_bad_rate = (
            pd.Series(target).mean() * 100
        )

        data = results_percentil.copy()

        data["bad_rate_aux"] = (
            data["bad_rate"] * 100
        )

        fig, ax = plt.subplots(
            figsize=self.figsize
        )

        sns.lineplot(
            data=data,
            x="percentile",
            y="bad_rate_aux",
            hue="model",
            palette=self.palette,
            ax=ax,
        )

        ax.axhline(
            y=avg_bad_rate,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=(
                f"Taxa de mau geral: "
                f"{avg_bad_rate/100:.1%}"
            ),
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        ax.legend()

        self._apply_font_sizes(ax)

        self._save_figure(
            fig,
            output_path,
        )

        return fig, ax

    def plot_lift(
        self,
        results_percentil: pd.DataFrame,
        title: Optional[str] = None,
        xlabel: str = "Decil",
        ylabel: str = "Lift",
        output_path: Optional[str] = None,
    ) -> FigAx:
        """
        Gráfico de lift por percentil.
        """

        self._validate_results_percentil(
            results_percentil,
            ["percentile", "lift", "model"],
        )

        fig, ax = plt.subplots(
            figsize=self.figsize
        )

        sns.lineplot(
            data=results_percentil,
            x="percentile",
            y="lift",
            hue="model",
            palette=self.palette,
            ax=ax,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        ax.legend()

        self._apply_font_sizes(ax)

        self._save_figure(
            fig,
            output_path,
        )

        return fig, ax

    def plot_gain(
        self,
        results_percentil: pd.DataFrame,
        title: Optional[str] = None,
        xlabel: str = "Decil",
        ylabel: str = "Ganho acumulado",
        output_path: Optional[str] = None,
    ) -> FigAx:
        """
        Gráfico de ganho acumulado.
        """

        self._validate_results_percentil(
            results_percentil,
            ["percentile", "gain", "model"],
        )

        data = results_percentil.copy()

        data["gain_aux"] = (
            data["gain"] * 100
        )

        zeros = pd.DataFrame([
            {
                "percentile": 0,
                "gain_aux": 0,
                "model": model,
            }
            for model in data["model"].unique()
        ])

        data = (
            pd.concat(
                [data, zeros],
                ignore_index=True,
            )
            .sort_values(
                ["model", "percentile"]
            )
            .reset_index(drop=True)
        )

        fig, ax = plt.subplots(
            figsize=self.figsize
        )

        sns.lineplot(
            data=data,
            x="percentile",
            y="gain_aux",
            hue="model",
            palette=self.palette,
            ax=ax,
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        ax.legend()

        self._apply_font_sizes(ax)

        self._save_figure(
            fig,
            output_path,
        )

        return fig, ax

    def plot_bad_rate_cum(
        self,
        results_percentil: pd.DataFrame,
        n_percentiles: int = 10,
        title: Optional[str] = None,
        deslocamento = 0,
        xlabel: str = "Decil",
        ylabel: str = "Taxa de mau acumulada",
        output_path: Optional[str] = None,
    ) -> FigAx:
        """
        Gráfico de taxa de mau acumulada reversa.
        """

        self._validate_results_percentil(
            results_percentil,
            [
                "percentile",
                "bad_rate_cum_rev",
                "model",
            ],
        )

        data = results_percentil.copy()

        data["bad_rate_cum_rev_aux"] = (
            data["bad_rate_cum_rev"] * 100
        )

        cutoff = self._compute_cutoff(
            n_percentiles
        )

        cutoff_data = data[
            data["percentile"] == cutoff
        ][
            [
                "model",
                "bad_rate_cum_rev_aux",
            ]
        ]

        fig, ax = plt.subplots(
            figsize=self.figsize
        )

        sns.lineplot(
            data=data,
            x="percentile",
            y="bad_rate_cum_rev_aux",
            hue="model",
            palette=self.palette,
            ax=ax,
        )

        ax.axvline(
            x=cutoff,
            color="gray",
            linestyle="--",
            linewidth=1,
            label=f"Corte no decil {cutoff}",
        )

        for _, row in cutoff_data.iterrows():

            value = row[
                "bad_rate_cum_rev_aux"
            ]

            ax.annotate(
                f"{value/100:.2%}",
                xy=(cutoff, value-deslocamento),
                xytext=(5, 0),
                textcoords="offset points",
                ha="left",
                va="center",
                fontsize=self.tick_fontsize,
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        if title is not None:
            ax.set_title(title)

        ax.invert_xaxis()

        ax.legend()

        self._apply_font_sizes(ax)

        self._save_figure(
            fig,
            output_path,
        )

        return fig, ax
