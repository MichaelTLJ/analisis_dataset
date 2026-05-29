import pandas as pd


REDSHIFT_BINS = [-float("inf"), 0.1, 0.5, 1.5, float("inf")]
REDSHIFT_LABELS = [
    "Redshift bajo",
    "Redshift medio",
    "Redshift alto",
    "Redshift extremo",
]


def add_science_groups(frame):
    grouped = frame.copy()
    grouped["REDSHIFT_GROUP"] = pd.cut(
        grouped["Z"],
        bins=REDSHIFT_BINS,
        labels=REDSHIFT_LABELS,
        include_lowest=True,
    ).astype(str)
    grouped["GROUP_ID"] = grouped["CLASS"] + " | " + grouped["REDSHIFT_GROUP"]
    return grouped


def build_treemap(frame):
    rows = (
        frame.groupby(["CLASS", "REDSHIFT_GROUP"], observed=False)
        .size()
        .reset_index(name="count")
        .sort_values(["CLASS", "REDSHIFT_GROUP"])
    )

    children = []
    for class_name, class_group in rows.groupby("CLASS", sort=False):
        class_children = [
            {
                "name": row["REDSHIFT_GROUP"],
                "groupId": f"{class_name} | {row['REDSHIFT_GROUP']}",
                "value": int(row["count"]),
            }
            for _, row in class_group.iterrows()
            if int(row["count"]) > 0
        ]
        children.append(
            {
                "name": class_name,
                "value": int(class_group["count"].sum()),
                "children": class_children,
            }
        )

    return {
        "name": "Objetos SDSS",
        "children": children,
    }

