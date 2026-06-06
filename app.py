"""
app.py — GA Task Assignment (v2)
Jalankan: streamlit run app.py
"""

import random
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from ga_engine import (
    GAConfig, GAResult, TaskAssignmentProblem,
    fitness, generate_problem, run_ga,
)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GA Task Assignment",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="metric-container"] {
    background: var(--background-color);
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 12px 16px;
}
.section-head {
    font-size: 14px; font-weight: 600; color: #374151;
    border-bottom: 2px solid #e5e7eb;
    padding-bottom: 4px; margin: 16px 0 10px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
for k, v in {
    "problem"       : None,
    "result"        : None,
    "running"       : False,
    "stop_req"      : False,
    "live_best"     : [],
    "live_avg"      : [],
    "live_feasible" : [],
    "live_gen"      : 0,
    "live_eval"     : None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 GA Task Assignment")
    st.caption("Extended Task Assignment Problem dengan Genetic Algorithm")
    st.divider()

    st.markdown("### ⚙️ Problem")
    n = st.slider("Jumlah karyawan / tugas (n)", 4, 20, 9)

    st.markdown("**Demand pengunjung per shift:**")
    c1, c2, c3 = st.columns(3)
    vp = c1.number_input("Pagi",  100, 2000, 100, step=50)
    vs = c2.number_input("Siang", 100, 2000, 200, step=50)
    vr = c3.number_input("Sore",  100, 2000, 500, step=50)
    visitor_demand = {'pagi': int(vp), 'siang': int(vs), 'sore': int(vr)}
    V_total = sum(visitor_demand.values())
    st.caption(f"Total: {V_total} pengunjung")

    st.markdown("**Jam kerja per shift:**")
    j1, j2, j3 = st.columns(3)
    jp = j1.number_input("Pagi",  1, 8,  3, key="jp")
    js = j2.number_input("Siang", 1, 8,  4, key="js")
    jr = j3.number_input("Sore",  1, 12, 5, key="jr")
    jam_shift = {'pagi': int(jp), 'siang': int(js), 'sore': int(jr)}

    col_a, col_b = st.columns(2)
    cost_min = col_a.number_input("Biaya min", 1, 20, 5)
    cost_max = col_b.number_input("Biaya max", 6, 50, 25)
    jam_min  = col_a.number_input("Jam min",   1, 8,  4)
    jam_max  = col_b.number_input("Jam max",   3, 14, 10)
    seed     = st.number_input("Seed", 0, 9999, 42)

    st.divider()
    st.markdown("### 🧬 Parameter GA")
    pop_size   = st.slider("Ukuran populasi",  20, 300, 120, step=10)
    max_gen    = st.slider("Maks generasi",    50, 1000, 400, step=50)
    alpha      = st.slider("Alpha (α)",        10, 200, 70,  step=5)
    mut_rate   = st.slider("Mutation rate",    1,  40,  12) / 100
    flip_rate  = st.slider("Shift flip rate",  1,  40,  12) / 100
    early_stop = st.slider("Early stop",       20, 200, 100, step=10)

    st.divider()
    if st.button("🔄 Generate Problem", width="stretch"):
        prob = generate_problem(
            n=n, visitor_demand=visitor_demand,
            cost_range=(int(cost_min), int(cost_max)),
            max_jam_range=(int(jam_min), int(jam_max)),
            seed=int(seed),
        )
        prob.jam_shift = jam_shift
        st.session_state.problem = prob
        st.session_state.result  = None
        st.session_state.live_best = []
        st.session_state.live_avg  = []
        st.session_state.live_feasible = []
        st.session_state.live_eval = None
        st.toast("✅ Problem di-generate!", icon="✅")

# Auto-generate on first load
if st.session_state.problem is None:
    prob = generate_problem(
        n=n, visitor_demand=visitor_demand,
        cost_range=(int(cost_min), int(cost_max)),
        max_jam_range=(int(jam_min), int(jam_max)),
        seed=int(seed),
    )
    prob.jam_shift = jam_shift
    st.session_state.problem = prob

prob: TaskAssignmentProblem = st.session_state.problem


# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab_prob, tab_run, tab_result, tab_study, tab_bench = st.tabs([
    "📋 Problem",
    "▶️ Jalankan GA",
    "📊 Hasil",
    "🔬 Studi Parameter",
    "🏃 Benchmark",
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — Problem
# ══════════════════════════════════════════════════════════════
with tab_prob:
    st.markdown("### 📋 Detail Problem")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Karyawan / Tugas", prob.n)
    m2.metric("Total Pengunjung", prob.V_total)
    m3.metric("Jumlah Shift", len(prob.shifts))
    m4.metric("Shift", " | ".join(s.capitalize() for s in prob.shifts))

    # Ideal distribution
    st.markdown('<div class="section-head">Distribusi Ideal Karyawan</div>', unsafe_allow_html=True)
    ideal = prob.ideal_count()
    ic1, ic2, ic3 = st.columns(3)
    for col, (s, v) in zip([ic1, ic2, ic3], prob.visitor_demand.items()):
        pct = 100 * v / prob.V_total
        col.metric(
            f"Shift {s.capitalize()}",
            f"{ideal[s]} karyawan",
            f"{v} pengunjung ({pct:.1f}%)",
        )

    # Availability table
    st.markdown('<div class="section-head">Ketersediaan & Jam Kerja Karyawan</div>', unsafe_allow_html=True)
    rows = []
    for i in range(prob.n):
        av = prob.avail[i]
        rows.append({
            "Karyawan"      : f"K{i+1}",
            "Shift Tersedia": " | ".join(s.capitalize() for s in av),
            "Max Jam Kerja" : f"{prob.max_jam[i]}j",
            "Fleksibilitas" : "✅ Semua shift" if len(av) == 3
                               else ("⚠️ 2 shift" if len(av) == 2 else "🔒 1 shift"),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Cost matrix
    st.markdown('<div class="section-head">Matriks Biaya</div>', unsafe_allow_html=True)
    st.caption("Baris = karyawan, Kolom = tugas. Warna merah = biaya tinggi.")
    cost_df = pd.DataFrame(
        prob.cost.astype(int),
        index=[f"K{i+1}" for i in range(prob.n)],
        columns=[f"T{j+1}" for j in range(prob.n)],
    )
    st.dataframe(
        cost_df.style.background_gradient(cmap="YlOrRd", axis=None),
        width="stretch",
    )

    # Formula Vd
    st.markdown('<div class="section-head">Rumus Fitness</div>', unsafe_allow_html=True)
    st.markdown(r"""
$$\text{fitness} = \text{total\_cost} + \alpha \cdot \left(V_d + 2 \cdot V_{\text{avail}} + V_{\text{jam}}\right)$$

$$V_d = \frac{1}{V_{\text{total}}} \sum_t V_t \left|\frac{N_t}{N} - \frac{V_t}{V_{\text{total}}}\right|$$

**Feasible** $\iff V_{\text{avail}} = 0$ **dan** $V_{\text{jam}} = 0$
""")


# ══════════════════════════════════════════════════════════════
# TAB 2 — Jalankan GA
# ══════════════════════════════════════════════════════════════
with tab_run:
    st.markdown("### ▶️ Jalankan Genetic Algorithm")

    cb1, cb2, _ = st.columns([1, 1, 5])
    run_btn  = cb1.button("🚀 Mulai",  width="stretch",
                           disabled=st.session_state.running)
    stop_btn = cb2.button("⏹ Stop",   width="stretch",
                           disabled=not st.session_state.running)

    if stop_btn:
        st.session_state.stop_req = True

    # Metric placeholders
    pm1, pm2, pm3, pm4, pm5 = st.columns(5)
    ph_gen  = pm1.empty(); ph_fit  = pm2.empty()
    ph_cost = pm3.empty(); ph_pen  = pm4.empty(); ph_feas = pm5.empty()
    ph_stat = st.empty()
    ph_chart = st.empty()

    def update_live(gen, ev, hb, ha, hf):
        ph_gen.metric("Generasi", gen + 1)
        if ev:
            ph_fit.metric("Best Fitness",  f"{ev['fitness']:.2f}")
            ph_cost.metric("Total Biaya",  f"{ev['total_cost']:.0f}")
            ph_pen.metric("Penalty",       f"{ev['penalty']:.2f}")
            ph_feas.metric("Feasible",     "✅ Ya" if ev['feasible'] else "❌ Tidak")
        if hb:
            gens = list(range(1, len(hb) + 1))
            fig = make_subplots(rows=1, cols=2,
                subplot_titles=("Konvergensi Fitness", "Individu Feasible per Generasi"))
            fig.add_trace(go.Scatter(x=gens, y=hb, name="Best",
                line=dict(color="#1D9E75", width=2),
                fill="tozeroy", fillcolor="rgba(29,158,117,0.08)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=gens, y=ha, name="Avg",
                line=dict(color="#378ADD", width=1.5, dash="dash")), row=1, col=1)
            fig.add_trace(go.Scatter(x=gens, y=hf, name="Feasible",
                line=dict(color="#BA7517", width=2),
                fill="tozeroy", fillcolor="rgba(186,117,23,0.10)"), row=1, col=2)
            fig.add_hline(y=pop_size, line_dash="dot", line_color="#1D9E75",
                          opacity=0.5, row=1, col=2)
            fig.update_layout(height=320, margin=dict(t=40, b=20),
                              legend=dict(orientation="h", y=1.12))
            fig.update_xaxes(title_text="Generasi")
            fig.update_yaxes(title_text="Fitness", col=1)
            fig.update_yaxes(title_text="Individu Feasible", col=2)
            ph_chart.plotly_chart(fig, width="stretch",
                                  key=f"live_{gen}")

    if run_btn:
        prob = st.session_state.problem
        cfg  = GAConfig(
            pop_size        = pop_size,
            max_gen         = max_gen,
            mut_rate        = mut_rate,
            shift_flip_rate = flip_rate,
            alpha           = float(alpha),
            early_stop      = early_stop,
        )
        prob.jam_shift = jam_shift

        st.session_state.running   = True
        st.session_state.stop_req  = False
        st.session_state.live_best = []
        ph_stat.info("⏳ GA sedang berjalan...")

        result = run_ga(
            prob=prob, config=cfg,
            stop_flag=lambda: st.session_state.stop_req,
            callback=lambda g, ev, hb, ha, hf: (
                update_live(g, ev, hb, ha, hf),
                st.session_state.update({
                    "live_best": hb, "live_avg": ha,
                    "live_feasible": hf, "live_gen": g, "live_eval": ev,
                }),
            )[0],
        )

        st.session_state.result  = result
        st.session_state.running = False

        update_live(
            result.generations_run - 1,
            result.best_eval,
            result.history_best,
            result.history_avg,
            result.history_feasible,
        )

        ev = result.best_eval
        msg = (
            f"✅ Selesai **{result.elapsed_time:.2f}s** | "
            f"**{result.generations_run}** generasi | "
            f"Fitness **{result.best_eval['fitness']:.2f}** | "
            f"Biaya **{int(ev['total_cost'])}** | "
            f"Vd **{ev['Vd']:.4f}** | "
            f"{'✅ Feasible' if ev['feasible'] else '❌ Infeasible'}"
        )
        if result.converged_at is not None:
            msg += f" | Konvergensi gen **{result.converged_at + 1}**"
        ph_stat.success(msg)
        st.toast("GA selesai!", icon="🎉")

    elif st.session_state.live_best:
        update_live(
            st.session_state.live_gen,
            st.session_state.live_eval,
            st.session_state.live_best,
            st.session_state.live_avg,
            st.session_state.live_feasible,
        )


# ══════════════════════════════════════════════════════════════
# TAB 3 — Hasil
# ══════════════════════════════════════════════════════════════
with tab_result:
    result = st.session_state.result
    prob   = st.session_state.problem

    if result is None:
        st.info("Belum ada hasil. Jalankan GA di tab **▶️ Jalankan GA**.")
    else:
        ev = result.best_eval
        st.markdown("### 📊 Hasil Terbaik")

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Total Biaya",     int(ev['total_cost']))
        r2.metric("Vd (proporsi)",   f"{ev['Vd']:.4f}")
        r3.metric("Penalty",         f"{ev['penalty']:.2f}")
        r4.metric("Generasi",        result.generations_run)
        r5.metric("Status", "✅ Feasible" if ev['feasible'] else "❌ Infeasible")

        # Constraint check
        st.markdown('<div class="section-head">Ringkasan Constraint</div>',
                    unsafe_allow_html=True)
        cc = st.columns(4)
        cc[0].metric("Avail. Violation", ev['avail_viol'],
                     delta="OK" if ev['avail_viol']==0 else "Ada pelanggaran",
                     delta_color="normal" if ev['avail_viol']==0 else "inverse")
        cc[1].metric("Jam Violation",    ev['jam_viol'],
                     delta="OK" if ev['jam_viol']==0 else "Ada pelanggaran",
                     delta_color="normal" if ev['jam_viol']==0 else "inverse")
        cc[2].metric("Waktu (s)", f"{result.elapsed_time:.2f}")
        if result.converged_at is not None:
            cc[3].metric("Feasible pertama gen", result.converged_at + 1)

        # Distribusi karyawan vs ideal
        st.markdown('<div class="section-head">Distribusi Karyawan vs Ideal</div>',
                    unsafe_allow_html=True)
        dist_rows = []
        for s in prob.shifts:
            Nt    = ev['shift_count'].get(s, 0)
            ideal = ev['ideal_count'].get(s, 0)
            dist_rows.append({
                "Shift"               : s.capitalize(),
                "Pengunjung (Vt)"     : prob.visitor_demand[s],
                "Proporsi Pengunjung" : f"{prob.visitor_demand[s]/prob.V_total:.3f}",
                "Karyawan Aktual (Nt)": Nt,
                "Proporsi Karyawan"   : f"{Nt/prob.n:.3f}",
                "Ideal"               : ideal,
                "Selisih"             : round(Nt - ideal, 2),
            })
        st.dataframe(pd.DataFrame(dist_rows), width="stretch", hide_index=True)

        # Assignment table
        st.markdown('<div class="section-head">Tabel Assignment</div>',
                    unsafe_allow_html=True)
        arows = []
        for i, (task, shift) in enumerate(result.best_chromosome):
            avail_ok = shift in prob.avail[i]
            jam_used = prob.jam_shift.get(shift, 0)
            jam_ok   = jam_used <= prob.max_jam[i]
            arows.append({
                "Karyawan"    : f"K{i+1}",
                "Tugas"       : f"T{task+1}",
                "Shift"       : shift.capitalize(),
                "Biaya"       : int(prob.cost[i][task]),
                "Availability": "✅ OK" if avail_ok else "❌ Melanggar",
                "Jam"         : f"{jam_used}j / {prob.max_jam[i]}j",
                "Jam OK"      : "✅" if jam_ok else "❌",
            })
        df_assign = pd.DataFrame(arows)

        def hl(row):
            ok = "✅ OK" in row["Availability"] and "✅" in row["Jam OK"]
            return [f"background-color: {'#f0faf5' if ok else '#fff5f5'}"] * len(row)

        st.dataframe(df_assign.style.apply(hl, axis=1),
                     width="stretch", hide_index=True)

        # Visualisasi
        st.markdown('<div class="section-head">Visualisasi</div>', unsafe_allow_html=True)
        vc1, vc2 = st.columns(2)

        with vc1:
            # Heatmap
            shapes = []
            for i, (task, shift) in enumerate(result.best_chromosome):
                ok = shift in prob.avail[i]
                shapes.append(dict(
                    type="rect",
                    x0=task-.5, x1=task+.5, y0=i-.5, y1=i+.5,
                    line=dict(color="#1D9E75" if ok else "#E24B4A", width=3),
                    fillcolor="rgba(0,0,0,0)",
                ))
            fig_h = go.Figure(go.Heatmap(
                z=prob.cost.tolist(),
                x=[f"T{j+1}" for j in range(prob.n)],
                y=[f"K{i+1}" for i in range(prob.n)],
                colorscale="YlOrRd",
                text=[[str(int(prob.cost[i][j])) for j in range(prob.n)]
                      for i in range(prob.n)],
                texttemplate="%{text}",
            ))
            fig_h.update_layout(title="Matriks Biaya (kotak = terpilih)",
                                height=380, margin=dict(t=40,b=20), shapes=shapes)
            st.plotly_chart(fig_h, width="stretch")

        with vc2:
            # Proporsi bar
            shifts = prob.shifts
            V_total = prob.V_total
            prop_vis = [prob.visitor_demand[s] / V_total for s in shifts]
            prop_emp = [ev['shift_count'].get(s, 0) / prob.n  for s in shifts]

            fig_p = go.Figure()
            fig_p.add_trace(go.Bar(
                x=[s.capitalize() for s in shifts], y=prop_vis,
                name="Proporsi Pengunjung", marker_color="#E24B4A", opacity=0.8,
                text=[f"{v:.3f}" for v in prop_vis], textposition="outside",
            ))
            fig_p.add_trace(go.Bar(
                x=[s.capitalize() for s in shifts], y=prop_emp,
                name="Proporsi Karyawan", marker_color="#378ADD", opacity=0.8,
                text=[f"{v:.3f}" for v in prop_emp], textposition="outside",
            ))
            fig_p.update_layout(
                title=f"Proporsi Karyawan vs Pengunjung (Vd={ev['Vd']:.4f})",
                barmode="group", height=380, margin=dict(t=40, b=20),
                legend=dict(orientation="h", y=1.12),
            )
            st.plotly_chart(fig_p, width="stretch")

        # Download
        st.download_button(
            "⬇️ Download Assignment (CSV)",
            data=df_assign.to_csv(index=False),
            file_name="ga_assignment.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════
# TAB 4 — Studi Parameter
# ══════════════════════════════════════════════════════════════
with tab_study:
    st.markdown("### 🔬 Studi Pengaruh Parameter")
    st.caption("Bandingkan pengaruh **alpha** dan **mutation rate** terhadap kualitas solusi.")

    prob_s = st.session_state.problem

    sc1, sc2 = st.columns(2)
    study_param  = sc1.selectbox("Parameter", ["Alpha (α)", "Mutation rate"])
    n_runs_study = sc1.slider("Run per nilai", 2, 8, 3)
    study_gen    = sc2.slider("Generasi (studi)", 50, 300, 200, step=25)
    study_pop    = sc2.slider("Populasi (studi)", 40, 150, 80,  step=10)

    if study_param == "Alpha (α)":
        param_values = [10, 30, 60, 100, 150]
        param_key    = "alpha"
    else:
        param_values = [0.02, 0.05, 0.10, 0.20, 0.30]
        param_key    = "mut"

    if st.button("▶️ Jalankan Studi", key="btn_study"):
        records = []
        total   = len(param_values) * n_runs_study
        bar     = st.progress(0, text="Memulai...")

        for vi, val in enumerate(param_values):
            fits, costs, Vds, feasibles = [], [], [], []
            for ri in range(n_runs_study):
                random.seed(ri * 13); np.random.seed(ri * 13)
                cfg_s = GAConfig(
                    pop_size        = study_pop,
                    max_gen         = study_gen,
                    alpha           = float(val) if param_key == "alpha" else float(alpha),
                    mut_rate        = float(val) if param_key == "mut"   else mut_rate,
                    shift_flip_rate = float(val) if param_key == "mut"   else flip_rate,
                    early_stop      = 60,
                )
                r = run_ga(prob_s, cfg_s)
                fits.append(r.best_eval['fitness'])
                costs.append(r.best_eval['total_cost'])
                Vds.append(r.best_eval['Vd'])
                feasibles.append(int(r.best_eval['feasible']))
                done = vi * n_runs_study + ri + 1
                bar.progress(done / total, text=f"Run {done}/{total}")

            records.append({
                study_param   : val,
                "Avg Fitness" : round(np.mean(fits), 2),
                "Std Fitness" : round(np.std(fits),  2),
                "Avg Biaya"   : round(np.mean(costs), 1),
                "Avg Vd"      : round(np.mean(Vds),   4),
                "Feasibility %": round(np.mean(feasibles) * 100, 0),
            })

        bar.empty()
        df_s = pd.DataFrame(records)
        st.dataframe(df_s, width="stretch", hide_index=True)

        x_vals = [str(v) for v in param_values]
        fig_s = make_subplots(rows=2, cols=2, subplot_titles=[
            f"Pengaruh {study_param} terhadap Fitness",
            f"Pengaruh {study_param} terhadap Feasibility",
            f"Pengaruh {study_param} terhadap Avg Biaya",
            f"Pengaruh {study_param} terhadap Avg Vd",
        ])
        fig_s.add_trace(go.Scatter(x=x_vals, y=df_s["Avg Fitness"],
            error_y=dict(type="data", array=df_s["Std Fitness"].tolist()),
            mode="lines+markers", line=dict(color="#1D9E75", width=2), marker=dict(size=8),
            name="Avg Fitness"), row=1, col=1)
        fig_s.add_trace(go.Bar(x=x_vals, y=df_s["Feasibility %"],
            marker_color="#378ADD", opacity=0.8, name="Feasibility %"), row=1, col=2)
        fig_s.add_trace(go.Scatter(x=x_vals, y=df_s["Avg Biaya"],
            mode="lines+markers", line=dict(color="#E24B4A", width=2), marker=dict(size=8),
            name="Avg Biaya"), row=2, col=1)
        fig_s.add_trace(go.Scatter(x=x_vals, y=df_s["Avg Vd"],
            mode="lines+markers", line=dict(color="#BA7517", width=2), marker=dict(size=8),
            name="Avg Vd"), row=2, col=2)
        fig_s.update_layout(height=560, margin=dict(t=40, b=20), showlegend=False)
        fig_s.update_xaxes(title_text=study_param)
        st.plotly_chart(fig_s, width="stretch")

        st.download_button(
            "⬇️ Download Hasil Studi (CSV)",
            data=df_s.to_csv(index=False),
            file_name="ga_study.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════
# TAB 5 — Benchmark
# ══════════════════════════════════════════════════════════════
with tab_bench:
    st.markdown("### 🏃 Multi-Run Benchmark")
    st.caption("Jalankan GA beberapa kali dengan seed berbeda untuk mengukur **konsistensi** solusi.")

    prob_b  = st.session_state.problem
    n_runs  = st.slider("Jumlah run", 3, 20, 10)

    if st.button("▶️ Jalankan Benchmark", key="btn_bench"):
        records_b = []
        bar_b     = st.progress(0, text="Memulai benchmark...")
        cfg_b     = GAConfig(
            pop_size=pop_size, max_gen=max_gen,
            mut_rate=mut_rate, shift_flip_rate=flip_rate,
            alpha=float(alpha), early_stop=early_stop,
        )

        for run in range(n_runs):
            random.seed(run * 7); np.random.seed(run * 7)
            r = run_ga(prob_b, cfg_b)
            ev_b = r.best_eval
            records_b.append({
                "Run"        : run + 1,
                "Fitness"    : round(r.best_eval['fitness'], 2),
                "Biaya"      : int(ev_b['total_cost']),
                "Vd"         : round(ev_b['Vd'], 4),
                "Penalty"    : round(ev_b['penalty'], 2),
                "Feasible"   : "✅" if ev_b['feasible'] else "❌",
                "Generasi"   : r.generations_run,
                "Waktu (s)"  : round(r.elapsed_time, 2),
            })
            bar_b.progress((run + 1) / n_runs, text=f"Run {run+1}/{n_runs}")

        bar_b.empty()
        df_b = pd.DataFrame(records_b)

        # Statistik
        feas_n = sum(1 for r in records_b if r["Feasible"] == "✅")
        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Feasibility", f"{feas_n}/{n_runs} ({100*feas_n//n_runs}%)")
        s2.metric("Best Fitness", df_b["Fitness"].min())
        s3.metric("Mean Fitness",
                  f"{df_b['Fitness'].mean():.1f} ± {df_b['Fitness'].std():.1f}")
        s4.metric("Mean Biaya",   f"{df_b['Biaya'].mean():.1f}")
        s5.metric("Avg Waktu",    f"{df_b['Waktu (s)'].mean():.2f}s")

        st.dataframe(df_b, width="stretch", hide_index=True)

        # Box + scatter
        fig_b = make_subplots(rows=1, cols=2,
            subplot_titles=("Distribusi Fitness per Run", "Biaya & Vd per Run"))
        fig_b.add_trace(go.Box(y=df_b["Fitness"], name="Fitness",
            marker_color="#1D9E75", boxpoints="all", jitter=0.3), row=1, col=1)
        fig_b.add_trace(go.Scatter(
            x=df_b["Run"], y=df_b["Biaya"],
            mode="lines+markers", name="Biaya",
            line=dict(color="#378ADD", width=2), marker=dict(size=7)), row=1, col=2)
        fig_b.add_trace(go.Scatter(
            x=df_b["Run"], y=df_b["Vd"] * 1000,
            mode="lines+markers", name="Vd ×1000",
            line=dict(color="#BA7517", width=1.5, dash="dot"), marker=dict(size=6)),
            row=1, col=2)
        fig_b.update_layout(height=340, margin=dict(t=40, b=20),
                            legend=dict(orientation="h", y=1.12))
        fig_b.update_xaxes(title_text="Run", col=2)
        st.plotly_chart(fig_b, width="stretch")

        st.download_button(
            "⬇️ Download Hasil Benchmark (CSV)",
            data=df_b.to_csv(index=False),
            file_name="ga_benchmark.csv",
            mime="text/csv",
        )
