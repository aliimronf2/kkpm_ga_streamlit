# 🧬 GA Task Assignment App (v2)

Aplikasi Streamlit untuk **Extended Task Assignment Problem** menggunakan Genetic Algorithm.

## Cara Menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

Buka browser di `http://localhost:8501`

## Struktur

```
ga_app_v2/
├── app.py          # UI Streamlit (5 tab)
├── ga_engine.py    # Core GA — persis dari notebook v2
├── requirements.txt
└── README.md
```

## Tab

| Tab | Isi |
|-----|-----|
| 📋 Problem | Matriks biaya, availability, rumus fitness |
| ▶️ Jalankan GA | Run GA dengan grafik konvergensi live |
| 📊 Hasil | Assignment terbaik, heatmap, proporsi Vd |
| 🔬 Studi Parameter | Pengaruh alpha & mutation rate |
| 🏃 Benchmark | Multi-run konsistensi |

## Rumus Fitness

```
fitness = total_cost + α × (Vd + 2×avail_viol + jam_viol)

Vd = (Σ_t Vt |Nt/N - Vt/V_total|) / V_total

Feasible ⟺ avail_viol = 0 DAN jam_viol = 0
```
