""" FEATURES """

MEDICATIONS = [
    "MD652 | FABIFLU TABLETS",
    "MD656 IV-BECT 6MG (ivermectin)",
    "5042D | VEKLURY",
    "9547D | PAXLOVID",
    "LAGEVRIO",
    "00584 | PYRIDOXIN LÉČIVA INJ",
    "24836 | ACIDUM ASCORBICUM BBP",
    "24814 | CALCIFEROL BBP 7,5 MG/ML",
    "00498 | MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK",
    "00449 | EREVIT 300 MG/ML",
    "89145 | VITAMIN C-INJEKTOPAS",
    "92973 ALPHA D3",
    "02963 | PREDNISON 20 LÉČIVA",
    "00269 | PREDNISON 5 LÉČIVA",
    "84090 | DEXAMED 6",
    "1275C | DEXAMETAZÓN KRKA",
    "MD661 BIODEXONE-DEXAMETHASONE",
    "2410B HYDROCORTISONE",
    "3242C | OLUMIANT 4 MG",
    "Anakinra",
    "RoActemra",
    "34045 | POLYOXIDONIUM 6 MG",
    "87299 | IMUNOR",
    "56930 IMMODIN",
    "Isoprinosine, ",
    "3879d INOMED",
    "35715 Azithromycin",
    "45954 Ceftriaxon",
    "0471B MOLOXIN",
    "9819A MOXIFLOXACIN",
    "58730 CIPROFLOXACIN KABI 200",
    "58746 CIPROFLOXACINKABI 400",
    "05044 OZZION",
    "4147C OMEMYL",
    "89662 NOLPAZA",
    "39397 PANTOPRAZOL",
    "62916 SMECTA",
    "30639 REASEC",
    "84370 LAGOSA",
    "93105 DEGAN ",
    "94918 AMBROBENE",
    "24859 PENTOXYPHILLINUM",
    "8893 ACC INJEKT",
    "24949 CODEIN ",
    "26846 OXANTIL",
    "FRAXIPARIN",
    "CLEXANE",
    "FRAGMIN",
    "ASPIRIN",
    "ANOPYRIN",
]
DISEASES = [
    "Hypertenzia",
    "Diabetes mellitus",
    "Kardiovaskulárne ochorenia",
    "Chronické respiračné ochorenia",
    "Renálne ochorenia",
    "Pečeňové ochorenia",
    "Onkologické ochorenia",
    "Imunosupresia",
]

LABEL_COLUMN = "Závažnosť priebehu ochorenia"
CATEGORICAL_FEATURES = [
    "Pohlavie",
    "Fajčenie",
    "Alkohol",
    "Vakcinácia",
    "Typ vakcíny",
    "A04.7",
    *DISEASES,
    *MEDICATIONS,
]

""" FILES AND PATHS """
SOURCE_DIR = "../processed_data"
FIRST_WAVE_FILE = "1. vlna všetko new 13.11.2024.xlsx_transformed.xlsx"
SECOND_WAVE_FILE = "2. vlna všetko new 13.11.2024.xlsx_transformed.xlsx"
THIRD_WAVE_FILE = "3. vlna všetko new 13.11.2024.xlsx_transformed.xlsx"
FOURTH_WAVE_FILE = "4. vlna do 31.5.2024 všetko new.xlsx_transformed.xlsx"
COMBINED_WAVES_FILE = "merged.xlsx"
