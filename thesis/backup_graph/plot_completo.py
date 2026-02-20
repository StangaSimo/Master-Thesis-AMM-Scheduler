import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re
import os
import numpy as np

# ==========================================
# CONFIGURAZIONE GLOBALE
# ==========================================
CSV_DIR = "csv/"
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ==========================================
# 1. CARICAMENTO DATI GENERALE
# ==========================================

def load_data_main():
    all_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    df_list = []
    
    print(f"[Main] Loading files from {CSV_DIR}...")

    for filename in all_files:
        try:
            df = pd.read_csv(filename)
            fname = os.path.basename(filename)
            
            # Default values
            matrix_size = 0
            is_hetero = False
            logic_label = "Unknown"
            batch_size = -1 # N/A

            # --- PARSING FILENAME ---
            
            # 1. Detect Size (Sxxxx)
            size_match = re.search(r'S(\d+)', fname)
            if size_match:
                matrix_size = int(size_match.group(1))
            
            # 2. Detect Batch (Bxx) - Solo per Static
            batch_match = re.search(r'B(\d+)', fname)
            if batch_match:
                batch_size = int(batch_match.group(1))

            # 3. Detect Logic Type
            if "static_part" in fname:
                logic_label = "Static Partitioning"
            elif "dynamic_homo" in fname:
                logic_label = "Dynamic Partitioning"
            elif "cuda_only_S" in fname:
                logic_label = "CUDA Only"
            
            # Hetero Logic
            elif "static_hetero" in fname:
                logic_label = "Static Partitioning"
                is_hetero = True
                matrix_size = -1 
            elif "dynamic_hetero" in fname:
                logic_label = "Dynamic Partitioning"
                is_hetero = True
                matrix_size = -1
            elif "cuda_only_hetero" in fname:
                logic_label = "CUDA Only"
                is_hetero = True
                matrix_size = -1
            else:
                continue # Skip large matrix or others

            # Aggiunta colonne
            df['Logic_Label'] = logic_label
            df['Matrix_Size'] = matrix_size
            df['Is_Hetero'] = is_hetero
            df['Batch_Size'] = batch_size
            
            # Calcolo Energia (CPU Pkg + GPU)
            pkg = df['Pkg_J'].fillna(0) if 'Pkg_J' in df.columns else 0
            gpu = df['NVGPU_J'].fillna(0) if 'NVGPU_J' in df.columns else 0
            df['Total_Energy_J'] = pkg + gpu

            df_list.append(df)
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not df_list:
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)

# ==========================================
# 2. SELEZIONE SPECIFICA (B20 Homo, B70 Hetero)
# ==========================================

def get_selected_static_vs_others(df):
    """
    Filtra la strategia Statica selezionando SOLO:
    - Batch 20 per Homogeneous
    - Batch 70 per Heterogeneous
    """
    # Separa Static dagli altri
    static_df = df[df['Logic_Label'] == "Static Partitioning"]
    others_df = df[df['Logic_Label'] != "Static Partitioning"]
    
    if static_df.empty:
        return others_df

    # Filtra Static Homogeneous (Is_Hetero == False) -> Batch 20
    static_homo = static_df[
        (static_df['Is_Hetero'] == False) & 
        (static_df['Batch_Size'] == 20)
    ]

    # Filtra Static Heterogeneous (Is_Hetero == True) -> Batch 70
    static_hetero = static_df[
        (static_df['Is_Hetero'] == True) & 
        (static_df['Batch_Size'] == 70)
    ]

    # Unisci i filtrati
    selected_static = pd.concat([static_homo, static_hetero])
    
    # Rinomina per coerenza nei grafici
    selected_static['Logic_Label'] = "Static Partitioning"
    
    return pd.concat([others_df, selected_static], ignore_index=True)

# ==========================================
# 3. PLOT: BATCH ANALYSIS (Mostra tutti i batch)
# ==========================================

def plot_batch_analysis(df):
    print("\n--- Generating Batch Size Comparisons ---")
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update({'figure.figsize': (14, 9), 'lines.linewidth': 3, 'lines.markersize': 9})

    static_df = df[df['Logic_Label'] == "Static Partitioning"]
    if static_df.empty: return

    # 1. Static Hetero Batch Comparison
    hetero_data = static_df[static_df['Is_Hetero'] == True]
    if not hetero_data.empty:
        plt.figure()
        sns.lineplot(
            data=hetero_data, x="Num_Tasks", y="Total_Time_ms", 
            hue="Batch_Size", marker="o", palette="viridis", linewidth=2.5
        )
        plt.title("Impact of Batch Size on Heterogeneous Workload")
        plt.xlabel("Number of Tasks") # UNIFORMATO
        plt.ylabel("Total Time (ms)")
        plt.legend(title="Batch Size", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig("analysis_batch_hetero.png", dpi=300)
        print("Saved: analysis_batch_hetero.png")
        plt.close()

    # 2. Static Homogeneous Batch Comparison
    sizes = sorted(static_df[(static_df['Is_Hetero'] == False)]['Matrix_Size'].unique())
    for size in sizes:
        subset = static_df[(static_df['Matrix_Size'] == size)]
        plt.figure()
        sns.lineplot(
            data=subset, x="Num_Tasks", y="Total_Time_ms", 
            hue="Batch_Size", marker="o", palette="viridis", linewidth=2.5
        )
        plt.title(f"Impact of Batch Size: Matrix {size}x{size}")
        plt.xlabel("Number of Tasks") # UNIFORMATO
        plt.ylabel("Total Time (ms)")
        plt.legend(title="Batch Size", bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"analysis_batch_S{size}.png", dpi=300)
        print(f"Saved: analysis_batch_S{size}.png")
        plt.close()

# ==========================================
# 4. PLOT: MAIN COMPARISONS (Time & Energy)
# ==========================================

def plot_main_comparisons(selected_df):
    print("\n--- Generating Main Comparisons (Time & Energy) ---")
    sns.set_theme(style="whitegrid", context="talk")
    palette_main = {
        "Dynamic Partitioning": "#3498db", "Static Partitioning": "#e74c3c", "CUDA Only": "#2ecc71"
    }
    metrics = [("Total_Time_ms", "Total Time (ms)", "time"), ("Total_Energy_J", "Total Energy (J)", "energy")]
    sizes = sorted(selected_df[selected_df['Is_Hetero'] == False]['Matrix_Size'].unique())

    for y_col, y_label, file_tag in metrics:
        # Homogeneous
        for size in sizes:
            subset = selected_df[(selected_df['Matrix_Size'] == size)]
            if subset.empty: continue
            plt.figure()
            sns.lineplot(
                data=subset, x="Num_Tasks", y=y_col, 
                hue="Logic_Label", style="Logic_Label", 
                markers=True, dashes=False, palette=palette_main
            )
            plt.title(f"{y_label} Comparison: Matrix {size}x{size}")
            plt.xlabel("Number of Tasks") # UNIFORMATO
            plt.ylabel(y_label)
            plt.legend(title="Logic")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"compare_{file_tag}_S{size}.png", dpi=300)
            print(f"Saved: compare_{file_tag}_S{size}.png")
            plt.close()

        # Heterogeneous
        hetero_subset = selected_df[selected_df['Is_Hetero'] == True]
        if not hetero_subset.empty:
            plt.figure()
            sns.lineplot(
                data=hetero_subset, x="Num_Tasks", y=y_col, 
                hue="Logic_Label", style="Logic_Label", 
                markers=True, dashes=False, palette=palette_main
            )
            plt.title(f"{y_label} Comparison: Heterogeneous Workload")
            plt.xlabel("Number of Tasks") # UNIFORMATO
            plt.ylabel(y_label)
            plt.legend(title="Logic")
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.savefig(f"compare_{file_tag}_Hetero.png", dpi=300)
            print(f"Saved: compare_{file_tag}_Hetero.png")
            plt.close()

# ==========================================
# 5. PLOT: TASK DISTRIBUTION (Stacked Bar)
# ==========================================

def plot_real_tasks(df, title, filename):
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'figure.figsize': (16, 8), 'font.size': 11})
    accel_colors = {"CUDA": "#76b900", "SYCL": "#0071c5", "OPENBLAS": "#e74c3c", "OPENVINO": "#f39c12"}

    task_cols = [c for c in df.columns if "_Tasks" in c and c != "Num_Tasks"]
    accel_map = {col: col.replace("_Tasks", "") for col in task_cols}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    strategies = ["Static Partitioning", "Dynamic Partitioning"]
    
    for i, strategy in enumerate(strategies):
        ax = axes[i]
        subset = df[df['Logic_Label'] == strategy].sort_values("Num_Tasks")
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        task_data = subset[task_cols].copy()
        task_data.columns = [accel_map[c] for c in task_data.columns]
        
        total_processed = task_data.sum(axis=1)
        task_data_pct = task_data.div(total_processed, axis=0).fillna(0) * 100
        task_data_pct["Num_Tasks"] = subset["Num_Tasks"].values
        
        my_colors = [accel_colors.get(col, "#333333") for col in task_data.columns if col != "Num_Tasks"]
        
        task_data_pct.plot(
            x="Num_Tasks", kind="bar", stacked=True, ax=ax, 
            color=my_colors, width=0.85, legend=False
        )
        
        ax.set_title(f"{strategy}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Tasks", fontsize=11) # UNIFORMATO
        ax.tick_params(axis='x', labelsize=8, rotation=45)
        
        if i == 0: ax.set_ylabel("Tasks Completed (%)", fontsize=11)
        else: ax.set_ylabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Accelerator", loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4)
    plt.suptitle(title, fontsize=16, weight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# ==========================================
# 6. PLOT: TIME DISTRIBUTION (Stacked Bar)
# ==========================================

def plot_time_distribution_chart(df, title, filename):
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'figure.figsize': (16, 8), 'font.size': 11})
    accel_colors = {"CUDA": "#76b900", "SYCL": "#0071c5", "OPENBLAS": "#e74c3c", "OPENVINO": "#f39c12"}

    work_cols = [c for c in df.columns if "_Tot_Work_ms" in c]
    accel_map = {}
    for col in work_cols:
        acc_name = col.replace("_Tot_Work_ms", "")
        accel_map[col] = acc_name

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    strategies = ["Static Partitioning", "Dynamic Partitioning"]
    
    for i, strategy in enumerate(strategies):
        ax = axes[i]
        subset = df[df['Logic_Label'] == strategy].sort_values("Num_Tasks")
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        time_data = subset[work_cols].copy()
        time_data.columns = [accel_map[c] for c in time_data.columns]
        
        total_work_time = time_data.sum(axis=1)
        time_data_pct = time_data.div(total_work_time, axis=0).fillna(0) * 100
        time_data_pct["Num_Tasks"] = subset["Num_Tasks"].values
        
        my_colors = [accel_colors.get(col, "#333333") for col in time_data.columns if col != "Num_Tasks"]
        
        time_data_pct.plot(
            x="Num_Tasks", kind="bar", stacked=True, ax=ax, 
            color=my_colors, width=0.85, legend=False
        )
        
        ax.set_title(f"{strategy}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Tasks", fontsize=11) # UNIFORMATO
        
        n_bars = len(time_data_pct)
        if n_bars > 25:
            step = 2
            ax.set_xticks(range(0, n_bars, step))
            ax.set_xticklabels(time_data_pct["Num_Tasks"].iloc[::step], rotation=45)
        else:
            ax.tick_params(axis='x', rotation=45)

        if i == 0: ax.set_ylabel("Workload Share (%)", fontsize=11)
        else: ax.set_ylabel("")
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 100)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Accelerator", loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)
    plt.suptitle(title, fontsize=16, weight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# ==========================================
# 7. LARGE MATRIX FUNCTIONS
# ==========================================

def plot_large_matrix_chunks():
    # Caricamento e plot
    all_files = glob.glob(os.path.join(CSV_DIR, "large_matrix*.csv"))
    data = []
    for filename in all_files:
        if "cuda" in filename: continue
        try:
            fname = os.path.basename(filename)
            match = re.search(r'large_matrix\s+(\d+)\s*\.csv', fname)
            if match:
                M = int(match.group(1))
                df = pd.read_csv(filename)
                row = {'M': M}
                task_cols = [c for c in df.columns if "_Tasks" in c and "Num_Tasks" not in c]
                for col in task_cols:
                    if "OPENVINO" in col: continue
                    row[col] = df[col].sum()
                data.append(row)
        except: pass
    if not data: return
    df_split = pd.DataFrame(data).sort_values('M')

    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'figure.figsize': (14, 8), 'font.size': 12})
    accel_colors = {"CUDA": "#76b900", "SYCL": "#0071c5", "OPENBLAS": "#e74c3c"}
    STEP = 8 
    df_sampled = df_split.iloc[::STEP].copy()

    plt.figure()
    task_cols = [c for c in df_sampled.columns if "_Tasks" in c]
    accel_map_tasks = {col: col.replace("_Tasks", "") for col in task_cols}
    plot_df = df_sampled.set_index('M')[task_cols]
    plot_df.columns = [accel_map_tasks[c] for c in plot_df.columns]
    colors_tasks = [accel_colors.get(l, "#333333") for l in plot_df.columns]
    
    ax = plot_df.plot(kind='bar', stacked=True, color=colors_tasks, width=0.85, figsize=(14, 8))
    plt.title("Sub-Tasks per Device (Large Matrix)", fontsize=16, fontweight='bold')
    plt.xlabel("Matrix Rows (M)")
    plt.ylabel("Count of Chunks")
    plt.legend(loc='upper left', ncol=3)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("large_matrix_chunks.png", dpi=300)
    print("Saved: large_matrix_chunks.png")
    plt.close()

def plot_large_matrix_analysis():
    # Load Split
    all_files = glob.glob(os.path.join(CSV_DIR, "large_matrix*.csv"))
    data = []
    for f in all_files:
        if "cuda" in f: continue
        try:
            match = re.search(r'large_matrix\s+(\d+)\s*\.csv', os.path.basename(f))
            if match:
                M = int(match.group(1))
                df = pd.read_csv(f)
                row = {'M': M}
                work_cols = [c for c in df.columns if "_Tot_Work_ms" in c]
                current_work_values = []
                for col in work_cols:
                    if "OPENVINO" in col: continue
                    val = df[col].sum()
                    row[col] = val
                    current_work_values.append(val)
                if current_work_values: row['Total_Time_ms'] = max(current_work_values)
                else: row['Total_Time_ms'] = 0
                data.append(row)
        except: pass
    if not data: return
    df_split = pd.DataFrame(data).sort_values('M')

    # Load CUDA
    df_cuda = pd.DataFrame()
    cuda_path = os.path.join(CSV_DIR, "large_matrix_cuda.csv")
    if os.path.exists(cuda_path):
        try:
            tmp = pd.read_csv(cuda_path)
            if 'CUDA_Tot_Work_ms' in tmp.columns: tmp['Total_Time_ms'] = tmp['CUDA_Tot_Work_ms']
            df_cuda = tmp.sort_values('M')
        except: pass

    # Plot
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'figure.figsize': (16, 7), 'font.size': 12})
    accel_colors = {"CUDA": "#76b900", "SYCL": "#0071c5", "OPENBLAS": "#e74c3c"}
    STEP = 5 
    df_split = df_split.iloc[::STEP]
    if not df_cuda.empty: df_cuda = df_cuda[df_cuda['M'].isin(df_split['M'])]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # Time
    sns.lineplot(data=df_split, x='M', y='Total_Time_ms', marker='o', markersize=7,
                 label='Split Strategy', color='#3498db', linewidth=3, ax=ax1)
    if not df_cuda.empty:
        sns.lineplot(data=df_cuda, x='M', y='Total_Time_ms', marker='X', markersize=7,
                     label='CUDA Only', color='#76b900', linewidth=2.5, linestyle='--', ax=ax1)
    ax1.set_title("Execution Time Comparison", fontsize=15, fontweight='bold')
    ax1.set_xlabel("Matrix Size", fontsize=12)
    ax1.set_ylabel("Total Time (ms)", fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Load
    work_cols = [c for c in df_split.columns if "_Tot_Work_ms" in c]
    accel_map = {col: col.replace("_Tot_Work_ms", "") for col in work_cols}
    plot_df = df_split[['M'] + work_cols].copy()
    plot_df.columns = ['M'] + [accel_map[c] for c in work_cols]
    accel_names = list(accel_map.values())
    plot_df['Total_Work'] = plot_df[accel_names].sum(axis=1)
    for acc in accel_names: plot_df[acc] = (plot_df[acc] / plot_df['Total_Work']) * 100
    
    bar_colors = [accel_colors.get(acc, "#333333") for acc in accel_names]
    plot_df.plot(x='M', y=accel_names, kind='bar', stacked=True, ax=ax2, color=bar_colors, width=0.9, legend=False)
    
    ax2.set_title("Load Balancing", fontsize=15, fontweight='bold')
    ax2.set_xlabel("Matrix Size", fontsize=12)
    ax2.set_ylabel("Workload Share (%)", fontsize=12)
    ax2.set_ylim(0, 100)
    
    n_bars = len(plot_df)
    if n_bars > 15:
        step = 2
        ax2.set_xticks(range(0, n_bars, step))
        ax2.set_xticklabels(plot_df['M'].iloc[::step], rotation=45)
    else: ax2.tick_params(axis='x', rotation=45)
    
    ax2.legend(title="Accelerator", loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("large_matrix_analysis.png", dpi=300, bbox_inches='tight')
    print("Saved: large_matrix_analysis.png")
    plt.close()

def plot_time_energy_clean():
    # Load
    all_files = glob.glob(os.path.join(CSV_DIR, "large_matrix*.csv"))
    data = []
    for f in all_files:
        try:
            fname = os.path.basename(f)
            df = pd.read_csv(f)
            if "large_matrix_cuda" in fname:
                tmp = pd.DataFrame()
                tmp['M'] = df['M']
                if 'CUDA_Tot_Work_ms' in df.columns: tmp['Total_Time_ms'] = df['CUDA_Tot_Work_ms']
                elif 'Total_Time_ms' in df.columns: tmp['Total_Time_ms'] = df['Total_Time_ms']
                if 'Pkg_J' in df.columns: tmp['Total_J'] = df['Pkg_J'].fillna(0) + df['NVGPU_J'].fillna(0)
                tmp['Strategy'] = 'CUDA Only'
                data.append(tmp)
                continue
            match = re.search(r'large_matrix\s+(\d+)\s*\.csv', fname)
            if match:
                M = int(match.group(1))
                work_cols = [c for c in df.columns if "_Tot_Work_ms" in c]
                t_time = max([df[c].sum() for c in work_cols]) if work_cols else 0
                t_j = 0
                if 'Pkg_J' in df.columns: t_j = df['Pkg_J'].sum() + df['NVGPU_J'].sum()
                data.append(pd.DataFrame([{'M': M, 'Total_Time_ms': t_time, 'Total_J': t_j, 'Strategy': 'Static Split Logic'}]))
        except: pass
    if not data: return
    df = pd.concat(data, ignore_index=True).sort_values('M')

    # Plot
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({'figure.figsize': (16, 7), 'font.size': 12})
    STEP = 5
    colors = {"Static Split Logic": "#3498db", "CUDA Only": "#76b900"}
    df_clean = df.iloc[::STEP]

    fig, (ax1, ax2) = plt.subplots(1, 2)
    sns.lineplot(data=df_clean, x='M', y='Total_Time_ms', hue='Strategy', palette=colors, style='Strategy', markers=True, dashes={'Static Split Logic':(1,0),'CUDA Only':(2,2)}, ax=ax1, linewidth=3, markersize=9)
    ax1.set_title("Execution Time", pad=15, fontweight='bold')
    ax1.set_xlabel("Matrix Rows (M)")
    ax1.set_ylabel("Total Time (ms)")
    ax1.grid(True, linestyle='--', alpha=0.6)

    sns.lineplot(data=df_clean, x='M', y='Total_J', hue='Strategy', palette=colors, style='Strategy', markers=True, dashes={'Static Split Logic':(1,0),'CUDA Only':(2,2)}, ax=ax2, linewidth=3, markersize=9)
    ax2.set_title("Energy Consumption", pad=15, fontweight='bold')
    ax2.set_xlabel("Matrix Rows (M)")
    ax2.set_ylabel("Total Energy (J)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig("large_matrix_time_energy_clean.png", dpi=300)
    print("Saved: large_matrix_time_energy_clean.png")
    plt.close()

# ==========================================
# 8. METRICHE
# ==========================================

def calculate_metrics_standard(df, title):
    print("\n" + "#"*60)
    print(f" {title}")
    print("#"*60)
    if df.empty:
        print(" -> Nessun dato trovato.")
        return
    pivot = df.pivot_table(index='Num_Tasks', columns='Label', values='Total_Time_ms', aggfunc='min').dropna()
    if pivot.empty or 'CUDA_Only' not in pivot.columns:
        print(" -> Dati insufficienti.")
        return

    print(f"\n[1] SPEEDUP ANALYSIS (Baseline: CUDA_Only)")
    if 'Dynamic' in pivot.columns:
        spd_dyn = pivot['CUDA_Only'] / pivot['Dynamic']
        print(f"   > Dynamic vs CUDA: Avg {spd_dyn.mean():.2f}x | Max {spd_dyn.max():.2f}x")
    if 'Static' in pivot.columns:
        spd_stat = pivot['CUDA_Only'] / pivot['Static']
        print(f"   > Static vs CUDA:  Avg {spd_stat.mean():.2f}x | Max {spd_stat.max():.2f}x")
    if 'Dynamic' in pivot.columns and 'Static' in pivot.columns:
        direct_spd = pivot['Static'] / pivot['Dynamic']
        diff = (direct_spd.mean() - 1) * 100
        print(f"   > Dynamic vs Static: Avg {direct_spd.mean():.2f}x ({diff:+.1f}%)")

    if 'Total_J' in df.columns:
        print(f"\n[2] ENERGY EFFICIENCY (Avg per Batch & Tasks/Joule)")
        # Calcolo dell'energia media
        en_stats = df.groupby('Label')['Total_J'].mean()
        
        # Calcolo dell'efficienza in Task/Joule riga per riga
        # Per evitare divisioni per zero
        df_safe = df[df['Total_J'] > 0].copy()
        df_safe['Tasks_per_Joule'] = df_safe['Num_Tasks'] / df_safe['Total_J']
        eff_stats = df_safe.groupby('Label')['Tasks_per_Joule'].mean()
        
        baseline_j = en_stats.get('CUDA_Only', 0)
        
        for strat in en_stats.index:
            val_j = en_stats[strat]
            val_eff = eff_stats.get(strat, 0)
            saving = ""
            if baseline_j > 0 and strat != 'CUDA_Only':
                pct = ((baseline_j - val_j) / baseline_j) * 100
                saving = f"({-pct:+.1f}% vs CUDA)"
                
            print(f"   > {strat:12s}: Avg Energy = {val_j:.2f} J {saving:15s} | Avg Efficiency = {val_eff:.2f} Tasks/J")

def calculate_metrics_large(df, title):
    print("\n" + "#"*60)
    print(f" {title}")
    print("#"*60)
    if df.empty:
        print(" -> Nessun dato trovato.")
        return
    pivot = df.pivot_table(index='M', columns='Label', values='Total_Time_ms').dropna()
    if 'Hybrid_Split' not in pivot.columns or 'CUDA_Only' not in pivot.columns:
        print(" -> Mancano dati.")
        return

    print(f"\n[1] SPEEDUP ANALYSIS")
    pivot['Speedup'] = pivot['CUDA_Only'] / pivot['Hybrid_Split']
    avg_spd = pivot['Speedup'].mean()
    max_spd = pivot['Speedup'].max()
    best_M = pivot['Speedup'].idxmax()
    print(f"   > Hybrid vs CUDA Only:")
    print(f"     Average Speedup: {avg_spd:.2f}x")
    print(f"     Max Speedup:     {max_spd:.2f}x (at M={best_M})")

    if 'Total_J' in df.columns:
        print(f"\n[2] ENERGY CONSUMPTION")
        en_pivot = df.pivot_table(index='M', columns='Label', values='Total_J').dropna()
        if not en_pivot.empty:
            en_pivot['Saving'] = (en_pivot['CUDA_Only'] - en_pivot['Hybrid_Split']) / en_pivot['CUDA_Only'] * 100
            avg_saving = en_pivot['Saving'].mean()
            print(f"   > Hybrid Energy Saving vs CUDA: {avg_saving:+.2f}% (Average)")
            total_cuda_j = en_pivot['CUDA_Only'].sum()
            total_hyb_j = en_pivot['Hybrid_Split'].sum()
            print(f"   > Total Energy used in Benchmark: CUDA={total_cuda_j:.0f}J vs Hybrid={total_hyb_j:.0f}J")

def load_data_metrics_filtered(size_str, batch_target):
    """Carica dati per metriche filtrando per batch specifico se Static"""
    all_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    data = []
    for f in all_files:
        if "large_matrix" in f or "hetero" in f: continue
        try:
            fname = os.path.basename(f)
            if size_str not in fname: continue
            df = pd.read_csv(f)
            label = "Unknown"
            if "static_part" in fname: label = "Static"
            elif "dynamic_homo" in fname: label = "Dynamic"
            elif "cuda_only" in fname: label = "CUDA_Only"
            
            if label == "Static":
                match = re.search(r'B(\d+)', fname)
                if match:
                    if int(match.group(1)) != batch_target: continue
                else: continue # Skip if no batch found

            if label != "Unknown":
                if 'Pkg_J' in df.columns: df['Total_J'] = df['Pkg_J'].fillna(0) + df['NVGPU_J'].fillna(0)
                cols = ['Num_Tasks', 'Total_Time_ms']
                if 'Total_J' in df.columns: cols.append('Total_J')
                subset = df[cols].copy()
                subset['Label'] = label
                data.append(subset)
        except: pass
    if not data: return pd.DataFrame()
    return pd.concat(data, ignore_index=True)

def load_hetero_metrics_filtered(batch_target):
    all_files = glob.glob(os.path.join(CSV_DIR, "*hetero*.csv"))
    data = []
    for f in all_files:
        try:
            fname = os.path.basename(f)
            df = pd.read_csv(f)
            label = "Unknown"
            if "static_hetero" in fname: label = "Static"
            elif "dynamic_hetero" in fname: label = "Dynamic"
            elif "cuda_only_hetero" in fname: label = "CUDA_Only"
            
            if label == "Static":
                match = re.search(r'B(\d+)', fname)
                if match:
                    if int(match.group(1)) != batch_target: continue
                else: continue

            if label != "Unknown":
                if 'Pkg_J' in df.columns: df['Total_J'] = df['Pkg_J'].fillna(0) + df['NVGPU_J'].fillna(0)
                cols = ['Num_Tasks', 'Total_Time_ms']
                if 'Total_J' in df.columns: cols.append('Total_J')
                subset = df[cols].copy()
                subset['Label'] = label
                data.append(subset)
        except: pass
    if not data: return pd.DataFrame()
    return pd.concat(data, ignore_index=True)

def load_large_matrix_metrics_data():
    all_files = glob.glob(os.path.join(CSV_DIR, "large_matrix*.csv"))
    data = []
    for f in all_files:
        try:
            fname = os.path.basename(f)
            df = pd.read_csv(f)
            if "large_matrix_cuda" in fname:
                subset = pd.DataFrame()
                subset['M'] = df['M']
                if 'CUDA_Tot_Work_ms' in df.columns: subset['Total_Time_ms'] = df['CUDA_Tot_Work_ms']
                elif 'Total_Time_ms' in df.columns: subset['Total_Time_ms'] = df['Total_Time_ms']
                if 'Pkg_J' in df.columns: subset['Total_J'] = df['Pkg_J'].fillna(0) + df['NVGPU_J'].fillna(0)
                subset['Label'] = 'CUDA_Only'
                data.append(subset)
                continue
            match = re.search(r'large_matrix\s+(\d+)\s*\.csv', fname)
            if match:
                M = int(match.group(1))
                work_cols = [c for c in df.columns if "_Tot_Work_ms" in c]
                t_time = max([df[c].sum() for c in work_cols]) if work_cols else 0
                t_j = 0
                if 'Pkg_J' in df.columns: t_j = df['Pkg_J'].sum() + df['NVGPU_J'].sum()
                data.append(pd.DataFrame([{'M': M, 'Total_Time_ms': t_time, 'Total_J': t_j, 'Label': 'Hybrid_Split'}]))
        except: pass
    if not data: return pd.DataFrame()
    return pd.concat(data, ignore_index=True)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=== AVVIO GENERAZIONE GRAFICI ===")

    # 1. Main Comparisons (Time & Energy)
    raw_df_main = load_data_main()
    if not raw_df_main.empty:
        plot_batch_analysis(raw_df_main) # Plot batch analysis shows ALL batches
        best_df_main = get_selected_static_vs_others(raw_df_main) # Selects B20/B70 for comparisons
        best_df_main = best_df_main.sort_values(by=['Matrix_Size', 'Num_Tasks'])
        plot_main_comparisons(best_df_main)

    # 2. Task Distribution (Real Tasks) & Time Distribution
    # Usiamo lo stesso dataframe filtrato "best_df_main" che ha già la selezione corretta (B20/B70)
    if not raw_df_main.empty:
        # Homogeneous
        sizes = sorted(best_df_main[(best_df_main['Is_Hetero'] == False)]['Matrix_Size'].unique())
        for size in sizes:
            subset = best_df_main[best_df_main['Matrix_Size'] == size]
            plot_real_tasks(subset, f"Task Distribution: Matrix {size}x{size}", f"distrib_tasks_S{size}.png")
            plot_time_distribution_chart(subset, f"Time Distribution: Matrix {size}x{size}", f"distrib_time_S{size}.png")
        
        # Heterogeneous
        hetero_subset = best_df_main[best_df_main['Is_Hetero'] == True]
        if not hetero_subset.empty:
            plot_real_tasks(hetero_subset, "Task Distribution: Heterogeneous", "distrib_tasks_Hetero.png")
            plot_time_distribution_chart(hetero_subset, "Time Distribution: Heterogeneous", "distrib_time_Hetero.png")

    # 3. Large Matrix Graphs
    plot_large_matrix_chunks()
    plot_large_matrix_analysis()
    plot_time_energy_clean()

    # 4. Metrics Calculation (Filtered B20 for Homo, B70 for Hetero)
    print("\n=== CALCOLO METRICHE ===")
    df_1024 = load_data_metrics_filtered("S1024", batch_target=20)
    calculate_metrics_standard(df_1024, "MATRIX 1024x1024 (Homogeneous - Static B20)")

    df_2048 = load_data_metrics_filtered("S2048", batch_target=20)
    calculate_metrics_standard(df_2048, "MATRIX 2048x2048 (Homogeneous - Static B20)")

    df_4096 = load_data_metrics_filtered("S4096", batch_target=20)
    calculate_metrics_standard(df_4096, "MATRIX 4096x4096 (Homogeneous - Static B20)")

    df_hetero_met = load_hetero_metrics_filtered(batch_target=70)
    calculate_metrics_standard(df_hetero_met, "HETEROGENEOUS WORKLOAD (Static B70)")

    df_large_met = load_large_matrix_metrics_data()
    calculate_metrics_large(df_large_met, "LARGE MATRIX SPLIT (Hybrid vs CUDA)")

    print("\n=== COMPLETATO TUTTO ===")
