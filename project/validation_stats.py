"""
validation_stats.py
===================
Validação estatística dos benchmarks de frameworks de Federated Learning
com base em múltiplas execuções com SEEDs distintos.

Frameworks avaliados : Flower, NVFlare, FedBioMed
Replicações          : seeds distintos (42, 69, 420) por configuração
Métricas analisadas  : Acurácia final/máxima, duração, GPU, RAM, rede

NOTA METODOLÓGICA:
  - A comparação GLOBAL entre frameworks é EXPLORATÓRIA: como cada framework
    rodou um subconjunto distinto de configurações, o pool global confunde
    efeito-de-framework com efeito-de-configuração. A análise PRINCIPAL é a
    comparação MATCHED/BLOCKED (mesma config para ≥2 frameworks).
  - A correção FDR de Benjamini-Hochberg é aplicada sobre TODA a família de
    testes pairwise (todas as métricas de uma vez), não por métrica isolada.
  - O p-value corrigido/estrelado é o do teste NÃO-PARAMÉTRICO (Mann-Whitney),
    coerente com a rejeição de normalidade (Shapiro-Wilk).
  - Réplicas duplicadas (mesmo config+seed re-executado) são desduplicadas
    mantendo a última Iteration, pois não são observações independentes.
"""

import json
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

for _msg in (".*p-value may be inaccurate.*", ".*Sample size too small.*",
             ".*Exact p-value calculation.*"):
    warnings.filterwarnings("ignore", message=_msg)

CONFIG_KEYS = ["Framework", "Param_Clients", "Param_Rounds",
               "Param_Epochs", "Param_Batch_Size"]

CONDITION_KEYS = ["Param_Clients", "Param_Rounds", "Param_Epochs", "Param_Batch_Size"]

REPLICATE_KEYS = CONFIG_KEYS + ["Param_Seed"]

# Métricas numéricas derivadas
NUMERIC_METRICS = [
    "final_accuracy",  # acurácia no último round  (%)
    "max_accuracy",  # acurácia máxima entre rounds  (%)
    "first_accuracy",  # acurácia no round 1  (%)
    "convergence_delta",  # max_accuracy − first_accuracy  (p.p.)
    "duration_s",  # duração da janela ajustada  (s)
    "server_gpu_util",  # utilização GPU servidor  (%)
    "clients_gpu_util",  # utilização GPU clientes — média  (%)
    "server_memory_gb",  # RAM servidor  (GB)
    "clients_memory_gb",  # RAM clientes — média  (GB)
    "server_net_bps",  # banda de rede servidor  Rx+Tx  (bps)
    "clients_net_bps",  # banda de rede clientes Rx+Tx médio  (bps)
]

# Métricas reportadas na comparação global (server_gpu_util omitido: sempre 0)
GLOBAL_REPORT_METRICS = [
    "final_accuracy", "max_accuracy", "duration_s",
    "clients_gpu_util", "server_memory_gb", "clients_memory_gb",
    "server_net_bps", "clients_net_bps",
]

# Threshold de CV para considerar alta variabilidade entre seeds
CV_HIGH_THRESHOLD = 0.15  # 15 %

# Nível de significância (alpha) para FDR e estrelas
ALPHA = 0.05


def _parse_accuracies(val) -> Optional[Dict[str, float]]:
    """Parse da coluna Accuracies_Per_Round (string JSON) para dict."""
    if pd.isna(val):
        return None
    try:
        return json.loads(str(val))
    except Exception:
        return None


def _round_number(round_name) -> Optional[int]:
    """Extrai o número do round de rótulos como 'Round 3', 'round_3', 'R3'."""
    import re
    m = re.search(r"(\d+)", str(round_name))
    return int(m.group(1)) if m else None


def _ordered_accuracies(d: Dict[str, float]) -> List[float]:
    """Acurácias ordenadas pelo número do round (não pela ordem do JSON)."""
    items = []
    for k, v in d.items():
        rn = _round_number(k)
        if rn is not None:
            items.append((rn, v))
    items.sort(key=lambda x: x[0])
    return [v for _, v in items]


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Correção FDR de Benjamini–Hochberg. Retorna q-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    if m == 0:
        return np.array([])
    order = np.argsort(pvals)
    ranked = pvals[order]
    bh = ranked * m / np.arange(1, m + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    qvals = np.empty_like(bh)
    qvals[order] = bh
    return np.clip(qvals, 0, 1)


def interpret_cohens_d(d: float) -> str:
    if d is None or np.isnan(d):
        return "N/A"
    a = abs(d)
    if a < 0.2:
        return "negligível"
    elif a < 0.5:
        return "pequeno"
    elif a < 0.8:
        return "médio"
    return "grande"


def sig_stars(p: float) -> str:
    if p is None or np.isnan(p):
        return "N/A"
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"


def pooled_std_weighted(g1: np.ndarray, g2: np.ndarray) -> float:
    """Desvio padrão pooled ponderado pelos graus de liberdade (n desbalanceado)."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    return np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))


class FLBenchmarkValidator:
    """
    Carrega os CSVs de benchmark, enriquece com métricas derivadas e
    expõe métodos de análise estatística para:

      1. Estatísticas descritivas por configuração
      2. Consistência cross-seed (replicabilidade)
      3. Normalidade das amostras por framework
      4. Comparação global entre frameworks (EXPLORATÓRIA)
      5. Comparação em configurações comuns / matched (PRINCIPAL)
      6. Detecção de outliers
      7. Análise de convergência por round
    """

    def __init__(self, csv_paths: List[str], dedup: bool = True):
        frames = []
        for p in csv_paths:
            try:
                frames.append(pd.read_csv(p))
            except FileNotFoundError:
                print(f" Arquivo não encontrado: {p} — ignorado.")
        if not frames:
            raise FileNotFoundError("Nenhum CSV encontrado.")

        self.raw = pd.concat(frames, ignore_index=True)
        self._dedup_replicates(enabled=dedup)
        self._enrich()

        fw_list = sorted(self.raw["Framework"].unique())
        n_configs = self.raw.groupby(CONFIG_KEYS).ngroups
        print(f" Dados carregados: {len(self.raw)} runs | "
              f"Frameworks: {fw_list} | Configurações únicas: {n_configs}\n")

    def _dedup_replicates(self, enabled: bool = True):
        """
        Réplicas com o MESMO (config + seed) não são observações independentes
        — geralmente são re-execuções. Mantê-las viola a premissa dos testes e
        infla artificialmente os n. Por padrão mantém a última Iteration.
        """
        dup_groups = self.raw.groupby(REPLICATE_KEYS).size()
        n_extra = int((dup_groups - 1).clip(lower=0).sum())
        if n_extra == 0:
            return

        print(f"{n_extra} runs duplicadas (mesmo config+seed) detectadas.")
        if not enabled:
            print("      → dedup DESATIVADA: duplicatas mantidas (use com cautela).\n")
            return

        if "Iteration" in self.raw.columns:
            self.raw = (self.raw.sort_values("Iteration")
                        .drop_duplicates(subset=REPLICATE_KEYS, keep="last")
                        .reset_index(drop=True))
        else:
            self.raw = self.raw.drop_duplicates(
                subset=REPLICATE_KEYS, keep="last").reset_index(drop=True)
        print(f"      → mantida a última Iteration de cada (config+seed). "
              f"Runs restantes: {len(self.raw)}.\n")

    def _enrich(self):
        """Extrai métricas derivadas e converte unidades."""
        df = self.raw

        df["_acc_dict"] = df["Accuracies_Per_Round"].apply(_parse_accuracies)
        df["_acc_ordered"] = df["_acc_dict"].apply(
            lambda d: _ordered_accuracies(d) if d else [])

        df["final_accuracy"] = df["_acc_ordered"].apply(
            lambda a: a[-1] * 100 if a else np.nan)
        df["max_accuracy"] = df["_acc_ordered"].apply(
            lambda a: max(a) * 100 if a else np.nan)
        df["first_accuracy"] = df["_acc_ordered"].apply(
            lambda a: a[0] * 100 if a else np.nan)
        df["convergence_delta"] = df["max_accuracy"] - df["first_accuracy"]

        df["duration_s"] = df["Adjusted_Window_s"]
        df["server_gpu_util"] = df["Server_GPU_Util"]
        df["clients_gpu_util"] = df["Clients_Avg_GPU_Util"]
        df["server_memory_gb"] = df["Server_Memory_Bytes"] / 1e9
        df["clients_memory_gb"] = df["Clients_Avg_Memory_Bytes"] / 1e9
        df["server_net_bps"] = df["Server_Net_Rx_Bps"] + df["Server_Net_Tx_Bps"]
        df["clients_net_bps"] = df["Clients_Avg_Net_Rx_Bps"] + df["Clients_Avg_Net_Tx_Bps"]

        self.raw = df

    def per_config_stats(self) -> pd.DataFrame:
        """
        Para cada (Framework + parâmetros), computa μ, σ, CV e IC 95 %
        sobre as réplicas com seeds distintos.
        """
        rows = []
        for keys, grp in self.raw.groupby(CONFIG_KEYS):
            row = dict(zip(CONFIG_KEYS, keys))
            row["n_seeds"] = len(grp)

            for metric in NUMERIC_METRICS:
                vals = grp[metric].dropna().values
                n = len(vals)
                if n == 0:
                    for sfx in ("mean", "std", "cv", "ci95_lo", "ci95_hi"):
                        row[f"{metric}_{sfx}"] = np.nan
                    continue
                m = vals.mean()
                s = vals.std(ddof=1) if n > 1 else 0.0
                cv = (s / m) if m != 0 else np.nan
                se = s / np.sqrt(n)
                t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
                row[f"{metric}_mean"] = round(m, 4)
                row[f"{metric}_std"] = round(s, 4)
                row[f"{metric}_cv"] = round(cv, 4) if not np.isnan(cv) else np.nan
                row[f"{metric}_ci95_lo"] = round(m - t_crit * se, 4)
                row[f"{metric}_ci95_hi"] = round(m + t_crit * se, 4)

            rows.append(row)

        return pd.DataFrame(rows)

    def cross_seed_consistency(self) -> Dict:
        """
        Verifica a consistência entre as réplicas (seeds) de cada configuração.
        Retorna dicionário com:
          - 'high_cv'  : configs com CV > threshold (alta variabilidade)
          - 'low_n'    : configs com < 3 seeds encontrados
          - 'summary'  : tabela completa de CV por config × métrica
        """
        results: Dict[str, list] = {"high_cv": [], "low_n": [], "summary": []}

        for keys, grp in self.raw.groupby(CONFIG_KEYS):
            cfg = dict(zip(CONFIG_KEYS, keys))
            n = len(grp)

            if n < 3:
                results["low_n"].append({**cfg, "n_found": n})

            for metric in ["final_accuracy", "duration_s", "clients_gpu_util"]:
                vals = grp[metric].dropna().values
                if len(vals) < 2:
                    continue
                m = vals.mean()
                s = vals.std(ddof=1)
                cv = (s / m) if m != 0 else np.nan  # padronizado com per_config_stats

                entry = {**cfg, "metric": metric,
                         "mean": round(m, 4), "std": round(s, 4),
                         "cv": round(cv, 4) if not np.isnan(cv) else np.nan,
                         "n": len(vals)}
                results["summary"].append(entry)

                if not np.isnan(cv) and cv > CV_HIGH_THRESHOLD:
                    results["high_cv"].append(entry)

        return results

    def normality_tests(self, metric: str = "final_accuracy") -> pd.DataFrame:
        """
        Aplica Shapiro-Wilk a cada framework para a métrica informada
        usando o pool de todas as réplicas do framework.
        """
        rows = []
        for fw, grp in self.raw.groupby("Framework"):
            vals = grp[metric].dropna().values
            if len(vals) < 3:
                rows.append({"Framework": fw, "metric": metric,
                             "n": len(vals), "W": np.nan, "p": np.nan, "normal": "N/A"})
                continue
            W, p = stats.shapiro(vals)
            rows.append({
                "Framework": fw, "metric": metric,
                "n": len(vals), "W": round(W, 4), "p": round(p, 4),
                "normal": "Sim" if p > 0.05 else "Não",
            })
        return pd.DataFrame(rows)

    def compare_frameworks(self, metric: str = "final_accuracy") -> pd.DataFrame:
        """
        Para a métrica indicada (SEM correção FDR — feita globalmente depois):
          - ANOVA one-way (F) e Kruskal-Wallis (H) entre frameworks
          - Comparações pairwise: Welch t-test + Mann-Whitney U
          - Cohen's d (tamanho do efeito, pooled ponderado)

        ATENÇÃO: análise exploratória. Pool global confunde framework com
        configuração (desenho desbalanceado). Ver matched_comparison.
        """
        frameworks = sorted(self.raw["Framework"].unique())
        groups: Dict[str, np.ndarray] = {
            fw: self.raw[self.raw["Framework"] == fw][metric].dropna().values
            for fw in frameworks
        }

        valid = {fw: v for fw, v in groups.items() if len(v) >= 2}
        if len(valid) < 2:
            return pd.DataFrame()

        # Se todos os valores forem idênticos (ex: server_gpu_util = 0 em todos)
        all_vals = np.concatenate(list(valid.values()))
        if np.all(all_vals == all_vals[0]):
            return pd.DataFrame()

        f_stat, anova_p = stats.f_oneway(*valid.values())
        try:
            h_stat, kruskal_p = stats.kruskal(*valid.values())
        except ValueError:
            h_stat, kruskal_p = np.nan, np.nan

        results = []
        valid_fws = sorted(valid.keys())
        pairs = [(a, b) for i, a in enumerate(valid_fws)
                 for b in valid_fws[i + 1:]]

        for fw1, fw2 in pairs:
            g1, g2 = valid[fw1], valid[fw2]

            t_stat, t_p = stats.ttest_ind(g1, g2, equal_var=False)  # Welch
            u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")

            mean_diff = g1.mean() - g2.mean()
            psd = pooled_std_weighted(g1, g2)
            cohens_d = mean_diff / psd if psd and not np.isnan(psd) and psd != 0 else np.nan

            results.append({
                "metric": metric,
                "framework_1": fw1,
                "framework_2": fw2,
                "n_fw1": len(g1),
                "n_fw2": len(g2),
                "mean_fw1": round(g1.mean(), 4),
                "mean_fw2": round(g2.mean(), 4),
                "mean_diff": round(mean_diff, 4),
                "f_stat_global": round(f_stat, 4),
                "anova_p_global": round(anova_p, 6),
                "h_stat_global": round(h_stat, 4) if not np.isnan(h_stat) else np.nan,
                "kruskal_p_global": round(kruskal_p, 6) if not np.isnan(kruskal_p) else np.nan,
                "t_stat": round(t_stat, 4),
                "t_p": round(t_p, 6),
                "u_stat": round(u_stat, 4),
                "u_p": round(u_p, 6),  # ← p-value principal (não-paramétrico)
                "cohens_d": round(cohens_d, 4) if not np.isnan(cohens_d) else np.nan,
                "effect_size": interpret_cohens_d(cohens_d),
            })

        return pd.DataFrame(results)

    def compare_frameworks_all_metrics(self) -> pd.DataFrame:
        """
        Executa compare_frameworks para todas as métricas e aplica a correção
        FDR de Benjamini-Hochberg sobre TODA a família de testes pairwise de
        uma só vez (não por métrica isolada).

        O p-value corrigido é o do teste NÃO-PARAMÉTRICO (Mann-Whitney U),
        coerente com a rejeição de normalidade.
        """
        frames = [self.compare_frameworks(m) for m in NUMERIC_METRICS]
        non_empty = [f for f in frames if not f.empty]
        if not non_empty:
            return pd.DataFrame()

        combined = pd.concat(non_empty, ignore_index=True)

        qvals = benjamini_hochberg(combined["u_p"].values)
        combined["q_fdr"] = np.round(qvals, 6)
        combined["sig_fdr"] = [sig_stars(q) for q in qvals]

        return combined

    def matched_comparison(self, metric: str = "final_accuracy") -> pd.DataFrame:
        """
        Identifica condições (Clients × Rounds × Epochs × Batch) com dados
        para pelo menos dois frameworks e aplica Kruskal-Wallis por bloco.
        FDR-BH aplicado sobre o conjunto de blocos.
        """
        frameworks = sorted(self.raw["Framework"].unique())
        all_conditions = (
            self.raw[CONDITION_KEYS].drop_duplicates()
            .apply(tuple, axis=1).tolist()
        )

        rows = []
        for cond in sorted(set(all_conditions)):
            cond_dict = dict(zip(CONDITION_KEYS, cond))
            mask_cond = pd.Series([True] * len(self.raw), index=self.raw.index)
            for k, v in cond_dict.items():
                mask_cond = mask_cond & (self.raw[k] == v)

            fw_vals: Dict[str, np.ndarray] = {}
            for fw in frameworks:
                vals = self.raw[mask_cond & (self.raw["Framework"] == fw)][metric].dropna().values
                if len(vals) > 0:
                    fw_vals[fw] = vals

            if len(fw_vals) < 2:
                continue

            row = {**cond_dict, "n_frameworks": len(fw_vals)}
            for fw in frameworks:
                if fw in fw_vals:
                    v = fw_vals[fw]
                    row[f"mean_{fw}"] = round(v.mean(), 4)
                    row[f"std_{fw}"] = round(v.std(ddof=1), 4) if len(v) > 1 else 0.0
                    row[f"n_seeds_{fw}"] = len(v)
                else:
                    row[f"mean_{fw}"] = np.nan
                    row[f"std_{fw}"] = np.nan
                    row[f"n_seeds_{fw}"] = 0

            try:
                pooled = np.concatenate(list(fw_vals.values()))
                if np.all(pooled == pooled[0]):
                    raise ValueError("sem variação")
                h, p = stats.kruskal(*fw_vals.values())
                row["kruskal_H"] = round(h, 4)
                row["kruskal_p"] = round(p, 6)
            except Exception:
                row["kruskal_H"] = np.nan
                row["kruskal_p"] = np.nan

            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # FDR-BH sobre os blocos com p-value válido
        valid_p = df["kruskal_p"].notna()
        df["q_fdr"] = np.nan
        df["sig"] = "N/A"
        if valid_p.any():
            q = benjamini_hochberg(df.loc[valid_p, "kruskal_p"].values)
            df.loc[valid_p, "q_fdr"] = np.round(q, 6)
            df.loc[valid_p, "sig"] = [sig_stars(x) for x in q]

        return df

    def detect_outliers(self) -> pd.DataFrame:
        """
        Dentro de cada configuração (Framework + parâmetros), detecta runs
        aberrantes via regra do IQR (1.5×) e |z-score| > 2.5.
        """
        outlier_rows = []

        for keys, grp in self.raw.groupby(CONFIG_KEYS):
            cfg = dict(zip(CONFIG_KEYS, keys))

            for metric in ["final_accuracy", "duration_s"]:
                vals = grp[metric].dropna()
                if len(vals) < 3:
                    continue

                Q1, Q3 = vals.quantile(0.25), vals.quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                if vals.values.std(ddof=0) == 0:
                    continue  # sem dispersão: z indefinido, nenhum outlier
                z_scores = np.abs(stats.zscore(vals.values))

                for idx, val, z in zip(vals.index, vals.values, z_scores):
                    iqr_flag = bool(val < lower or val > upper)
                    z_flag = bool(z > 2.5)
                    if iqr_flag or z_flag:
                        outlier_rows.append({
                            **cfg,
                            "Param_Seed": self.raw.loc[idx, "Param_Seed"],
                            "metric": metric,
                            "value": round(val, 4),
                            "z_score": round(z, 4),
                            "iqr_outlier": iqr_flag,
                            "z_outlier": z_flag,
                        })

        return pd.DataFrame(outlier_rows) if outlier_rows else pd.DataFrame()

    def convergence_analysis(self) -> pd.DataFrame:
        """
        Expande Accuracies_Per_Round em linhas
        (Framework × config × seed × round), útil para plotar curvas.
        """
        rows = []
        n_skipped = 0
        for _, r in self.raw.iterrows():
            d = r["_acc_dict"]
            if not d:
                continue
            for round_name, acc in d.items():
                round_num = _round_number(round_name)
                if round_num is None:
                    n_skipped += 1
                    continue
                rows.append({
                    "Framework": r["Framework"],
                    "Param_Clients": r["Param_Clients"],
                    "Param_Rounds": r["Param_Rounds"],
                    "Param_Epochs": r["Param_Epochs"],
                    "Param_Batch_Size": r["Param_Batch_Size"],
                    "Param_Seed": r["Param_Seed"],
                    "Round": round_num,
                    "Accuracy_pct": round(acc * 100, 4),
                })
        if n_skipped:
            print(f"   {n_skipped} rounds ignorados (rótulo sem número reconhecível).")
        return pd.DataFrame(rows)

    def generate_report(self) -> Dict[str, pd.DataFrame]:
        SEP = "=" * 80
        DIV = "-" * 80

        print(SEP)
        print("  RELATÓRIO DE VALIDAÇÃO ESTATÍSTICA — BENCHMARK FL FRAMEWORKS")
        print(SEP)

        print("\n1. VISÃO GERAL DOS DADOS")
        print(DIV)
        fw_counts = self.raw.groupby("Framework").size().reset_index(name="n_runs")
        print(fw_counts.to_string(index=False))

        seed_dist = (self.raw.groupby(["Framework", "Param_Seed"])
                     .size().reset_index(name="n_runs"))
        print("\n  Distribuição por framework × seed:")
        print(seed_dist.to_string(index=False))

        config_dist = (self.raw.groupby(CONFIG_KEYS).size()
                       .reset_index(name="n_seeds")
                       .sort_values(CONFIG_KEYS))
        print(f"\n  Configurações únicas: {len(config_dist)}")
        print("  (Framework, Clients, Rounds, Epochs, Batch → n_seeds)")
        print(config_dist.to_string(index=False))

        print(f"\n{SEP}")
        print("2. REPLICABILIDADE — CONSISTÊNCIA ENTRE SEEDS")
        print(DIV)
        cs = self.cross_seed_consistency()

        if cs["low_n"]:
            print(f"\n  ️  Configurações com menos de 3 seeds:")
            print(pd.DataFrame(cs["low_n"]).to_string(index=False))
        else:
            print("   Todas as configurações possuem ≥ 3 seeds")

        summary_df = pd.DataFrame(cs["summary"])
        if not summary_df.empty:
            print(f"\n  Resumo CV (coeficiente de variação) — threshold: {CV_HIGH_THRESHOLD:.0%}")
            for metric in summary_df["metric"].unique():
                sub = summary_df[summary_df["metric"] == metric]
                print(f"\n  Métrica: {metric}")
                agg = sub.groupby("Framework")["cv"].describe()[
                    ["count", "mean", "std", "min", "50%", "max"]
                ]
                print(agg.to_string())

        if cs["high_cv"]:
            print(f"\n    {len(cs['high_cv'])} configurações com CV > {CV_HIGH_THRESHOLD:.0%}:")
            print(pd.DataFrame(cs["high_cv"]).to_string(index=False))
        else:
            print(f"\n   Nenhuma configuração com CV > {CV_HIGH_THRESHOLD:.0%}")

        print(f"\n{SEP}")
        print("3. ESTATÍSTICAS DESCRITIVAS POR CONFIGURAÇÃO")
        print(DIV)
        cfg_stats = self.per_config_stats()
        display_cols = CONFIG_KEYS + [
            "n_seeds",
            "final_accuracy_mean", "final_accuracy_std", "final_accuracy_cv",
            "max_accuracy_mean",
            "duration_s_mean", "duration_s_std",
            "clients_gpu_util_mean",
        ]
        available = [c for c in display_cols if c in cfg_stats.columns]
        print(cfg_stats[available].to_string(index=False))

        print(f"\n{SEP}")
        print("4. TESTE DE NORMALIDADE (Shapiro-Wilk) — pool por framework")
        print(DIV)
        for metric in ["final_accuracy", "duration_s", "clients_gpu_util"]:
            norm_df = self.normality_tests(metric)
            print(f"\n  Métrica: {metric}")
            print(norm_df.to_string(index=False))

        print(f"\n{SEP}")
        print("5. COMPARAÇÃO GLOBAL ENTRE FRAMEWORKS  [EXPLORATÓRIA]")
        print(DIV)
        print("    Desenho desbalanceado: o pool global confunde efeito-de-")
        print("      framework com efeito-de-configuração. Trate como exploratória.")
        print("      A análise PRINCIPAL (matched/blocked).")
        print("      p-value corrigido (FDR-BH) = Mann-Whitney U, sobre TODAS as métricas.")

        all_comp = self.compare_frameworks_all_metrics()

        for metric in GLOBAL_REPORT_METRICS:
            sub = all_comp[all_comp["metric"] == metric] if not all_comp.empty else pd.DataFrame()
            if sub.empty:
                continue
            print(f"\n  ── {metric} ──")
            print(f"     ANOVA  : F={sub['f_stat_global'].iloc[0]:.4f}, "
                  f"p={sub['anova_p_global'].iloc[0]:.4f}  (paramétrico, referência)")
            kp = sub['kruskal_p_global'].iloc[0]
            print(f"     Kruskal: H={sub['h_stat_global'].iloc[0]:.4f}, "
                  f"p={kp:.4f}  (não-paramétrico, principal)")
            cols = ["framework_1", "framework_2",
                    "mean_fw1", "mean_fw2", "mean_diff",
                    "u_p", "q_fdr", "cohens_d", "effect_size", "sig_fdr"]
            print(sub[cols].to_string(index=False))

        if not all_comp.empty:
            print(f"\n  Tabela-resumo de significância (FDR-BH, Mann-Whitney) — todas as métricas:")
            all_comp["pair"] = all_comp["framework_1"] + " vs " + all_comp["framework_2"]
            pivot = all_comp.pivot_table(
                index="metric", columns="pair", values="sig_fdr", aggfunc="first"
            )
            print(pivot.to_string())

        print(f"\n{SEP}")
        print("6. COMPARAÇÃO EM CONFIGURAÇÕES COMUNS (MATCHED/BLOCKED)  [PRINCIPAL]")
        print(DIV)
        print("  Kruskal-Wallis intra-bloco (mesma config, ≥2 frameworks).")
        print("  Controla o confounder de configuração. FDR-BH sobre os blocos.")
        matched = self.matched_comparison("final_accuracy")
        if not matched.empty:
            n_testable = int(matched["kruskal_p"].notna().sum())
            print(f"\n  Configurações com ≥ 2 frameworks: {len(matched)} "
                  f"(testáveis: {n_testable})")
            print(matched.to_string(index=False))
            sig_count = int(matched["sig"].isin(["*", "**", "***"]).sum())
            print(f"\n  Com diferença significativa após FDR (q < 0.05): "
                  f"{sig_count}/{n_testable}")
        else:
            print("  ️  Nenhuma configuração com dados para múltiplos frameworks")

        print(f"\n{SEP}")
        print("7. DETECÇÃO DE OUTLIERS  (IQR × 1.5  |  |z| > 2.5)")
        print(DIV)
        outliers = self.detect_outliers()
        if not outliers.empty:
            print(f"  Total de runs outliers: {len(outliers)}")
            print(outliers.to_string(index=False))
        else:
            print("   Nenhum outlier detectado")

        print(f"\n{SEP}")
        print("8. ANÁLISE DE CONVERGÊNCIA — ACURÁCIA POR ROUND")
        print(DIV)
        conv = self.convergence_analysis()
        if not conv.empty:
            conv_summary = (
                conv.groupby(["Framework", "Round"])["Accuracy_pct"]
                .agg(mean="mean", std="std", n="count")
                .reset_index()
            )
            conv_summary["mean"] = conv_summary["mean"].round(4)
            conv_summary["std"] = conv_summary["std"].round(4)
            print(conv_summary.to_string(index=False))

            print("\n  Acurácia final — estatísticas por framework:")
            print(self.raw.groupby("Framework")["final_accuracy"]
                  .describe().round(4).to_string())

        print(f"\n{SEP}")
        print("  FIM DO RELATÓRIO")
        print(SEP)

        return {
            "config_stats": cfg_stats,
            "framework_comparison": all_comp,
            "matched_comparison": matched,
            "outliers": outliers,
            "convergence": conv,
            "cross_seed_summary": summary_df,
        }

def main():
    CSV_PATHS = [
        "../run_metrics.csv",
        "../run_metrics2.csv",
        "../run_metrics3.csv",
    ]

    validator = FLBenchmarkValidator(CSV_PATHS)
    results = validator.generate_report()

    output_files = {
        "stats_per_config.csv": results["config_stats"],
        "stats_framework_comparison.csv": results["framework_comparison"],
        "stats_cross_seed.csv": results["cross_seed_summary"],
        "stats_convergence.csv": results["convergence"],
    }
    if not results["matched_comparison"].empty:
        output_files["stats_matched_comparison.csv"] = results["matched_comparison"]
    if not results["outliers"].empty:
        output_files["stats_outliers.csv"] = results["outliers"]

    print("\n  Resultados salvos em:")
    for filename, df in output_files.items():
        df.to_csv(filename, index=False)
        print(f"    • {filename}")


if __name__ == "__main__":
    main()
