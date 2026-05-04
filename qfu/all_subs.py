QUARK_FLAVORS = ["up", "down", "charm", "strange", "top", "bottom"]

G_FIELDS=[
    "photon",  # A_μ
    "w_plus",  # W⁺
    "w_minus",  # W⁻
    "z_boson",  # Z⁰
    *[f"gluon_{i}" for i in range(8)]
]


H = [
    "higgs_field" # avoid call it higgs becaus moduel ahs same name
]

QUARKS = [f"{flav}_quark_{i}" for flav in QUARK_FLAVORS for i in range(3)]
FERMIONS=[
    # Leptonen
    "electron",  # ψₑ
    "muon",  # ψ_μ
    "tau",  # ψ_τ
    "electron_neutrino",  # νₑ
    "muon_neutrino",  # ν_μ
    "tau_neutrino",  # ν_τ
    *QUARKS
]



G_MAP=[
    "photon",  # A_μ
    "w_plus",  # W⁺
    "w_minus",  # W⁻
    "z_boson",  # Z⁰
]

GLON_MAP = [f"gluon_{i}" for i in range(8)]

ALL_SUBS_LOWER=[*FERMIONS, *G_FIELDS, *H]
ALL_SUBS=[*[f.upper() for f in FERMIONS],*[g.upper() for g in G_FIELDS],*[h.upper() for h in H]]

ALL_SUBS_DICT = {
    "FERMION": [f.upper() for f in FERMIONS],
    "GAUGE": [g.upper() for g in G_FIELDS],
    "HIGGS": [g.upper() for g in H],
}
