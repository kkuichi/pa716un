import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def categorize_severity(row: pd.Series) -> int | None:
    if row["Kód prepustenia"] in [1, 2, 5]:
        return 1
    if row["Kód prepustenia"] in [3, 4]:
        return 2
    if row["Kód prepustenia"] in [6, 7]:
        return 3
    return None


def fajcenie(row):
    row = str(row).lower()
    result = False
    if "fajč" in row or "cig" in row:
        result = True

    if (
        "fajčenie: 0" in row
        or "cigarety: 0" in row
        or "nefaj" in row
        or "fajčenie neg" in row
    ):
        result = False

    return result


def alkohol(row):
    row = str(row).lower()
    result = False
    if (
        "vínko" in row
        or "vino" in row
        or "víno" in row
        or "rum" in row
        or "pivo" in row
        or "pivko" in row
        or "alkohol" in row
        or "príležitostne" in row
    ):
        result = True

    if "alkohol: 0" in row or "alkohol ne" in row or "nepije" in row:
        result = False

    return result


def transform_test_results(
    column_value: str | None,
    row_index: int,
    column_name: str,
    errors_dict: dict[int, list[str]],
) -> tuple[float, float, float, float] | tuple[None, None, None, None]:
    try:
        column_value = column_value.replace("_x000D_", "")
    except Exception:
        return None, None, None, None

    try:
        # extract values
        column_value = column_value.strip()
        entries = column_value.split(";")
        values = [entry.split("|")[-1] for entry in entries if entry]
        values = [record.strip() for record in values]

        float_values = []
        skipped_values = []
        # convert to float
        for value in values:
            try:
                float_values.append(float(value.replace(",", ".")))
            except Exception:
                skipped_values.append(value)

        if not float_values:
            print(f"[INVALID] Row {row_index}: No valid values found in {column_value}")
            errors_dict[row_index].append(f"{column_name}")
            return None, None, None, None

        first = float_values[0]
        last = float_values[-1]
        min_val = min(float_values)
        max_val = max(float_values)

        return first, last, min_val, max_val
    except Exception:
        print(f"[ERROR] Row {row_index}: Error processing {column_value}")
        return None, None, None, None


def create_labs_column(
    df: pd.DataFrame, columns_to_transform: list[str]
) -> pd.DataFrame:
    errors_dict = {row_index: [] for row_index in range(len(df))}
    transformed_data = []

    for column in columns_to_transform:
        print(f"PROCESSING COLUMN: {column}")
        transformed_column = df.apply(
            lambda row: transform_test_results(
                row[column], row.name, column, errors_dict
            ),
            axis=1,
            result_type="expand",
        )
        transformed_column.columns = [
            f"{column} first",
            f"{column} last",
            f"{column} min",
            f"{column} max",
        ]
        transformed_data.append(transformed_column)
        df.drop(columns=[column], inplace=True)

    df = pd.concat([df] + transformed_data, axis=1)
    # df["Error detected"] = df.index.map(
    #     lambda x: "|".join(errors_dict[x]) if errors_dict[x] else None
    # )
    return df


def create_labs_column_2(
    df: pd.DataFrame, columns_to_transform: list[str]
) -> pd.DataFrame:
    errors_dict = {row_index: [] for row_index in range(len(df))}
    transformed_data = []

    for column in columns_to_transform:
        print(f"PROCESSING COLUMN: {column}")
        transformed_column = df.apply(
            lambda row: transform_test_results(
                row[column], row.name, column, errors_dict
            ),
            axis=1,
            result_type="expand",
        )
        transformed_column.columns = [
            f"{column} first",
            f"{column} last",
            f"{column} min",
            f"{column} max",
        ]
        transformed_data.append(transformed_column)
        # df.drop(columns=[column], inplace=True)

    df = pd.concat([df] + transformed_data, axis=1)
    # df["Error detected"] = df.index.map(
    #     lambda x: "|".join(errors_dict[x]) if errors_dict[x] else None
    # )
    return df


def _search_keywords(row, columns, keywords):
    for keyword in keywords:
        pattern = re.compile(keyword, re.IGNORECASE)
        for column in columns:
            for column_value in str(row[column]).lower().split(","):
                if pattern.search(column_value):
                    return True

    return False


def detect_vaccination(
    row,
    positive_keywords,
    negative_keywords,
    columns_to_check=None,
):
    if not columns_to_check:
        columns_to_check = ["Epikríza", "Epidemiologická anamnéza"]

    positive_result = _search_keywords(row, columns_to_check, positive_keywords)
    negative_result = _search_keywords(row, columns_to_check, negative_keywords)

    if negative_result:
        return False

    if positive_result:
        return True

    return False


def detect_keywords(row: pd.Series, keywords: list[str]) -> bool:
    columns_to_check = ["Osobná anamnéza", "Epikríza", "Diagnózy"]
    for keyword in keywords:
        for column in columns_to_check:
            if keyword.lower() in str(row[column]).lower():
                return True

    return False


def get_vaccination_type(
    row, columns_to_check=["Epikríza", "Epidemiologická anamnéza"]
):
    m_rna_keywords = ["mrna", "pfizer", "moderna", "biontech", "biontec", "comirnaty"]
    viral_vector_keywords = ["astrazeneca", "johnson", "janssen", "astra", "zeneca"]

    for column in columns_to_check:
        for keyword in m_rna_keywords:
            if keyword in str(row[column]).lower():
                return "mRNA"

        for keyword in viral_vector_keywords:
            if keyword in str(row[column]).lower():
                return "vector"

    return None


def get_vaccination_count(
    row, columns_to_check=["Epikríza", "Epidemiologická anamnéza"]
):
    keywords = [
        r"očkovanie \b(\d+)x",
        r"očkovan[ý|ie|á] (\d+)[x|\.]$",
        r"očkovan[ý|ie|á] (\d+)[x|\.]\s+",
        r"\b(\d+)\s?dávok",
        r"\b(\d+)\s?dávkov",
        r"\b(\d+)\s?dávky",
        r"\b(\d+)\s?dávka",
        r"\b(\d+)\s?x$",
        r"\b(\d+)\s?x\s+",
        r"\b(\d+)x očkov",
        r"\b(\d+)\.\s?dáv",
        r"očkov.+(\d+)krát",
        "(\d+). dávk",
    ]

    if row["Vakcinácia"] == False:
        return 0

    for column in columns_to_check:
        for keyword in keywords:
            for row_value in str(row[column]).lower().split(","):
                pattern = re.compile(keyword, re.IGNORECASE)
                match = re.search(pattern, row_value)
                if match:  # Sum up all captured numbers
                    result = int(match.group(1))
                    if result > 4:
                        return 1
                    return result

    return 1


def detect_medication(row: pd.Series, keywords: list[str]) -> bool:
    columns_to_check = ["Lieková anamnéza", "Liečba"]
    for keyword in keywords:
        for column in columns_to_check:
            if keyword.lower() in str(row[column]).lower():
                return True

    return False


def overcame_covid(row: pd.Series) -> bool:
    pattern_prekon = re.compile(r"\bprekon", re.IGNORECASE)
    negative_pattern = re.compile(r"\bprekonanie:\s*0", re.IGNORECASE)

    if negative_pattern.search(str(row["Epidemiologická anamnéza"])):
        return False
    elif pattern_prekon.search(str(row["Epidemiologická anamnéza"])):
        return True
    return False


def detect_a047(row: pd.Series) -> bool:
    pattern = re.compile(r"A04.7", re.IGNORECASE)
    return bool(pattern.search(str(row["Diagnózy"])))


def detect_satO2(row: pd.Series) -> float | None:
    pattern1 = r"(?:SpO2|sat(?:uruje)?|Sat O2|sO2|satO2|O2 pulzoxymetricky|kyslíka|saturácia|O2-|sat.|s02|02|Sat.O2|SpO2|O2|\d+l/min|O2 na|O2 VM|sat. bez kYdlíka|sat. bez klíka|oxymetrom|s2|bey)\.?:?\s*(\d+%?)"
    pattern2 = r"(\d+%)(?:\s*\d+l)"
    pattern3 = r"(\d+%)(?:\s*na \d+l)"
    pattern4 = r"(\d+%)(?:\s*sat)"
    columns_to_check = ["Objektívny nález", "Terajšie ochorenie", "Epikríza"]

    for column in columns_to_check:
        for pattern in [pattern1, pattern2, pattern3, pattern4]:
            match = re.search(pattern, str(row[column]), re.IGNORECASE)
            if match:
                return float(match.group(1).replace("%", ""))

    return None


def show_confusion_matrix(y_test, y_pred):
    # Vizualizácia maticu chýb (confusion matrix)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Zoznam tried (podľa unikátnych hodnôt v y_test)
    class_labels = np.unique(y_test)

    plt.figure(figsize=(6, 6))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Class {label}" for label in class_labels],
        yticklabels=[f"Class {label}" for label in class_labels],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.show()


def show_roc(y_test, y_pred_proba):
    # Kreslenie ROC krivky pre viacero tried
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_curve, auc

    class_labels = np.unique(y_test)

    # Binarizácia y_test pre viacero tried (One-vs-Rest)
    y_test_binarized = label_binarize(y_test, classes=class_labels)
    n_classes = len(class_labels)

    # Počet tried: y_pred_proba má tvar (n_samples, n_classes)
    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(8, 6))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(
            fpr[i], tpr[i], label=f"Class {class_labels[i]} (AUC = {roc_auc[i]:.2f})"
        )

    # Pridanie diagonály
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")

    plt.title("Multiclass ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.savefig("roc.png", dpi=300, bbox_inches="tight")
    plt.show()
    # confusion matrix


def show_metrics(y_test, y_pred, y_pred_proba):
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        classification_report,
    )

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, y_pred_proba, multi_class="ovr")

    # Výstup metrík
    print("Výsledky modelu:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print("\nKlasifikačná správa:")
    print(classification_report(y_test, y_pred))
