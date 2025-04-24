# pa716un

Tento repozitár obsahuje projekt zameraný na spracovanie a analýzu zdravotníckych dát pomocou interpretovateľných modelov strojového učenia. Projekt využíva metodológiu CRISP-DM a je implementovaný v jazyku Python s využitím Jupyter Notebookov a knižníc pre moderné modelovanie a vizualizáciu.

## 🧠 Ciele projektu

- Predikcia závažnosti priebehu ochorenia COVID-19 u hospitalizovaných pacientov
- Využitie interpretovateľných modelov ako EBM, CatBoost, LightGBM
- Aplikácia metód vysvetliteľnosti ako SHAP a vizualizácia výstupov

## 📁 Štruktúra projektu

```
.
├── core/                 # Pomocné triedy a konfigurácie
├── data_preparation/    # Skripty na načítanie a spracovanie vstupných dát
├── data_visualisation/  # Grafické výstupy a vizualizácie
├── modeling/            # Tréning a vyhodnotenie modelov
├── requirements.txt     # Zoznam závislostí
└── README.md            # Tento súbor
```

## ⚙️ Inštalácia

1. Klonujte repozitár:

```bash
git clone https://github.com/kkuichi/pa716un.git
cd pa716un
```

2. Vytvorte a aktivujte virtuálne prostredie:

```bash
python -m venv venv
source venv/bin/activate  # Na Windows: venv\Scripts\activate
```

3. Nainštalujte požadované balíky:

```bash
pip install -r requirements.txt
```

## 🚀 Spustenie

Projekt je určený na spustenie v prostredí Jupyter Notebook. Po aktivácii prostredia spustite:

```bash
jupyter notebook
```

a prechádzajte jednotlivé kroky spracovania dát, tréningu modelov a vyhodnotenia.

## 📊 Použité technológie

- Python 3.10+
- scikit-learn
- CatBoost
- LightGBM
- SHAP
- EBM (interpret)
- Pandas, Numpy
- Streamlit (prototyp nasadenia)

## 🧪 Ukážka výstupu

Model Explainable Boosting Machine dosiahol najvyššiu presnosť na dátach 4. vlny COVID-19 a poskytol zrozumiteľné výstupy pre klinickú interpretáciu, vrátane vizualizácie najdôležitejších atribútov.
