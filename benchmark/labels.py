"""Display labels and skip list for the benchmark plots."""

# Maps data keys → paper display notation (matplotlib mathtext / LaTeX)
DISPLAY_LABELS = {
    # ShapG variants (the paper uses the cosine and mutual-information ones)
    "shapg | rank_deletion_mixed": "ShapG",
    "shapg | kendalltau": r"ShapG$^*$",
    "shapg | rank_deletion_cosine": r"ShapG$_{\cos}$",
    "shapg | rank_deletion_kendalltau": r"ShapG$_{\tau}$",
    "shapg | rank_deletion_mutual_info": r"ShapG$_{MI}$",
    "cis": "CIS",
    # KernelSHAP/SamplingSHAP: shap library on a fixed model;
    # LeverageSHAP: on the retrain-based characteristic function
    "kernel_shap": "KernelSHAP",
    "sampling_shap": "SamplingSHAP",
    "leverage_shap": "LeverageSHAP",
    "markovian_empty": r"Markovian$_\emptyset$",
    "markovian_corr06": r"Markovian$_\rho$",
    "markovian_mi003": r"Markovian$_\rho$",
    "markovian_full": r"Markovian$_\mathcal{N}$",
    # Strategy x initial set labels
    "markovian_greedy_empty": r"Markovian$^G_\emptyset$",
    "markovian_greedy_rho": r"Markovian$^G_\rho$",
    "markovian_greedy_full": r"Markovian$^G_\mathcal{N}$",
    "markovian_linear_empty": r"Markovian$^L_\emptyset$",
    "markovian_linear_rho": r"Markovian$^L_\rho$",
    "markovian_linear_full": r"Markovian$^L_\mathcal{N}$",
    "markovian_softmax_01_empty": r"Markovian$^S_\emptyset$",
    "markovian_softmax_01_rho": r"Markovian$^S_\rho$",
    "markovian_softmax_01_full": r"Markovian$^S_\mathcal{N}$",
    "markovian_top_p_09_t01_empty": r"Markovian$^{Tp}_\emptyset$",
    "markovian_top_p_09_t01_rho": r"Markovian$^{Tp}_\rho$",
    "markovian_top_p_09_t01_full": r"Markovian$^{Tp}_\mathcal{N}$",
    "markovian_top_k_3_empty": r"Markovian$^{Tk}_\emptyset$",
    "markovian_top_k_3_rho": r"Markovian$^{Tk}_\rho$",
    "markovian_top_k_3_full": r"Markovian$^{Tk}_\mathcal{N}$",
    "markovian_min_p_05_t05_empty": r"Markovian$^{Mp}_\emptyset$",
    "markovian_min_p_05_t05_rho": r"Markovian$^{Mp}_\rho$",
    "markovian_min_p_05_t05_full": r"Markovian$^{Mp}_\mathcal{N}$",
    "Greedy Best": "Greedy Best",
}

# Methods to exclude from KPI / efficiency plots
SKIP_METHODS = {"Random", "Greedy Worst"}


def display(key):
    """Return the display label for a data key."""
    return DISPLAY_LABELS.get(key, key)
