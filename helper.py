import pandas as pd


def build_port_to_index(
    csv_path="port_visits_with_sequences_collapsed.csv", allowed_ports=None
):
    """
    Build a mapping from sequence tokens to discrete indices, **filtered by allowed_ports**.

    allowed_ports: list of ports in your environment (e.g., ["PORT HAWKESBURY", "PORT HASTINGS", ...])
    """
    df = pd.read_csv(csv_path)

    if "PORT_Sequence" in df.columns:
        tokens = set()
        for seq in df["PORT_Sequence"].dropna():
            tokens.update(x.strip() for x in str(seq).split(",") if x.strip())
    elif "H3_Sequence" in df.columns:
        tokens = set()
        for seq in df["H3_Sequence"].dropna():
            tokens.update(x.strip() for x in str(seq).split(",") if x.strip())
    else:
        raise ValueError("CSV must contain PORT_Sequence or H3_Sequence")

    if allowed_ports is not None:
        tokens = tokens.intersection(set(allowed_ports))

    return {tok: i for i, tok in enumerate(sorted(tokens))}
