"""Method labels and row order for the paper tables."""

# Maps data keys → paper display notation (matplotlib mathtext / LaTeX)
DISPLAY_LABELS = {
    # KernelSHAP/SamplingSHAP: shap library on a fixed model;
    # LeverageSHAP: on the retrain-based characteristic function
    "kernel_shap": "KernelSHAP",
    "sampling_shap": "SamplingSHAP",
    "leverage_shap": "LeverageSHAP",
    "shapg | rank_deletion_mixed": r"ShapG",
    "shapg | kendalltau": r"ShapG$^*$",
    "shapg | rank_deletion_cosine": r"ShapG$_{\cos}$",
    "shapg | rank_deletion_kendalltau": r"ShapG$^\dagger_{\tau}$",
    "shapg | rank_deletion_mutual_info": r"ShapG$_{MI}$",
    "cis": "CIS",
    "markovian_empty": r"MPE$_\emptyset$",
    "markovian_corr06": r"MPE$_\rho$",
    "markovian_mi003": r"MPE$_\rho$",
    "markovian_full": r"MPE$_\mathcal{N}$",
    # Strategy x initial set labels
    "markovian_greedy_empty": r"MPE$^G_\emptyset$",
    "markovian_greedy_rho": r"MPE$^G_\rho$",
    "markovian_greedy_full": r"MPE$^G_\mathcal{N}$",
    "markovian_linear_empty": r"MPE$^L_\emptyset$",
    "markovian_linear_rho": r"MPE$^L_\rho$",
    "markovian_linear_full": r"MPE$^L_\mathcal{N}$",
    "markovian_softmax_01_empty": r"MPE$^S_\emptyset$",
    "markovian_softmax_01_rho": r"MPE$^S_\rho$",
    "markovian_softmax_01_full": r"MPE$^S_\mathcal{N}$",
    "markovian_top_p_09_t01_empty": r"MPE$^{Tp}_\emptyset$",
    "markovian_top_p_09_t01_rho": r"MPE$^{Tp}_\rho$",
    "markovian_top_p_09_t01_full": r"MPE$^{Tp}_\mathcal{N}$",
    "markovian_top_k_3_empty": r"MPE$^{Tk}_\emptyset$",
    "markovian_top_k_3_rho": r"MPE$^{Tk}_\rho$",
    "markovian_top_k_3_full": r"MPE$^{Tk}_\mathcal{N}$",
    "markovian_min_p_05_t05_empty": r"MPE$^{Mp}_\emptyset$",
    "markovian_min_p_05_t05_rho": r"MPE$^{Mp}_\rho$",
    "markovian_min_p_05_t05_full": r"MPE$^{Mp}_\mathcal{N}$",
    "Greedy Best": "Greedy Best",
}

# Ordered method spec for table generation: (data_key, group)
# "__threshold__" is resolved per dataset to the appropriate threshold key.
METHOD_SPEC = [
    ("Greedy Best", "oracle"),
    ("kernel_shap", "shap_baseline"),
    ("sampling_shap", "shap_baseline"),
    ("leverage_shap", "shap_baseline"),
    ("shapg | rank_deletion_cosine", "shapg_cis"),
    ("shapg | rank_deletion_mutual_info", "shapg_cis"),
    ("markovian_empty", "markovian"),
    ("__threshold__", "markovian"),  # resolved per dataset
    ("markovian_full", "markovian"),
]


def display(key):
    """Return the display label for a data key."""
    return DISPLAY_LABELS.get(key, key)
