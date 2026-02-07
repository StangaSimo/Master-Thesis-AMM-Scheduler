import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import re
import os

# ==========================================
# CONFIGURAZIONE ESTETICA (Stile "Paper" Pulito)
# ==========================================
sns.set_theme(style="whitegrid", context="paper")

plt.rcParams.update({
    'figure.figsize': (16, 8),
    'font.size': 11,
    'axes.titlesize': 15,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 11
})

CSV_DIR = "csv/"

# Colori (OPENVINO INCLUSO)
ACCEL_COLORS = {
    "CUDA": "#76b900",      # Nvidia Green
    "SYCL": "#0071c5",      # Intel Blue
    "OPENBLAS": "#e74c3c",  # Red
    "OPENVINO": "#f39c12"   # Orange
}

def load_data():
    all_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    df_list = []
    
    print(f"Loading files from {CSV_DIR}...")

    for filename in all_files:
        try:
            # Saltiamo i file large matrix per questo grafico specifico
            if "large_matrix" in filename: continue
            
            df = pd.read_csv(filename)
            fname = os.path.basename(filename)
            
            matrix_size = 0
            is_hetero = False
            logic_label = "Unknown"
            
            # Detect Size
            size_match = re.search(r'S(\d+)', fname)
            if size_match:
                matrix_size = int(size_match.group(1))
            
            # Detect Logic
            if "static_part" in fname:
                logic_label = "Static"
            elif "dynamic_homo" in fname:
                logic_label = "Dynamic"
            elif "static_hetero" in fname:
                logic_label = "Static"
                is_hetero = True
                matrix_size = -1
            elif "dynamic_hetero" in fname:
                logic_label = "Dynamic"
                is_hetero = True
                matrix_size = -1
            else:
                continue 

            df['Logic_Label'] = logic_label
            df['Matrix_Size'] = matrix_size
            df['Is_Hetero'] = is_hetero
            
            df_list.append(df)
            
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    if not df_list: return pd.DataFrame()
    return pd.concat(df_list, ignore_index=True)

def get_best_static_and_dynamic(df):
    """
    Filtra: Tutto il Dynamic + Solo il 'Best Batch' Static (tempo minimo)
    """
    dynamic_df = df[df['Logic_Label'] == "Dynamic"]
    static_df = df[df['Logic_Label'] == "Static"]
    
    if not static_df.empty:
        idx = static_df.groupby(['Matrix_Size', 'Is_Hetero', 'Num_Tasks'])['Total_Time_ms'].idxmin()
        best_static = static_df.loc[idx]
    else:
        best_static = pd.DataFrame()

    return pd.concat([best_static, dynamic_df], ignore_index=True)

def plot_task_distribution_chart(df, title, filename):
    """
    Grafico Stacked Bar basato sul NUMERO DI TASK (Real Distribution).
    """
    # Identifica le colonne dei task (Escludendo il totale 'Num_Tasks')
    task_cols = [c for c in df.columns if "_Tasks" in c and c != "Num_Tasks"]
    
    # Mappa nomi colonne -> nomi acceleratori puliti
    accel_map = {}
    for col in task_cols:
        acc_name = col.replace("_Tasks", "")
        accel_map[col] = acc_name

    # Creiamo figura con 2 subplot (Static vs Dynamic)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    strategies = ["Static", "Dynamic"]
    
    for i, strategy in enumerate(strategies):
        ax = axes[i]
        subset = df[df['Logic_Label'] == strategy].sort_values("Num_Tasks")
        
        if subset.empty:
            ax.text(0.5, 0.5, "No Data", ha='center')
            continue

        # Estraiamo i dati TASK
        task_data = subset[task_cols].copy()
        task_data.columns = [accel_map[c] for c in task_data.columns]
        
        # Calcolo percentuale reale (Tasks Completed / Total Processed)
        total_processed = task_data.sum(axis=1)
        task_data_pct = task_data.div(total_processed, axis=0).fillna(0) * 100
        
        # Aggiungiamo Num_Tasks per l'asse X
        task_data_pct["Num_Tasks"] = subset["Num_Tasks"].values
        
        # Colori corretti (Usiamo la mappa definita in alto)
        my_colors = [ACCEL_COLORS.get(col, "#333333") for col in task_data.columns if col != "Num_Tasks"]
        
        # Plotting
        task_data_pct.plot(
            x="Num_Tasks", 
            kind="bar", 
            stacked=True, 
            ax=ax, 
            color=my_colors,
            width=0.85, 
            legend=False
        )
        
        ax.set_title(f"{strategy}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Total Tasks", fontsize=11)
        
        # Ottimizzazione Etichette X (Salto se troppe per evitare il muro di testo)
        n_bars = len(task_data_pct)
        if n_bars > 25:
            step = 2
            ax.set_xticks(range(0, n_bars, step))
            ax.set_xticklabels(task_data_pct["Num_Tasks"].iloc[::step], rotation=45)
        else:
            ax.tick_params(axis='x', rotation=45)

        if i == 0:
            ax.set_ylabel("Tasks Completed (%)", fontsize=11)
        else:
            ax.set_ylabel("")
        
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Legenda Unica
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Accelerator", loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True)
    
    plt.suptitle(title, fontsize=16, weight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    raw_df = load_data()
    
    if not raw_df.empty:
        clean_df = get_best_static_and_dynamic(raw_df)
        
        # 1. Grafici per Matrici Omogenee
        sizes = sorted(clean_df[(clean_df['Is_Hetero'] == False)]['Matrix_Size'].unique())
        for size in sizes:
            subset = clean_df[clean_df['Matrix_Size'] == size]
            plot_task_distribution_chart(
                subset, 
                title=f"Task Distribution: Matrix {size}x{size}", 
                filename=f"distrib_tasks_S{size}.png"
            )
            
        # 2. Grafico Eterogeneo
        hetero_subset = clean_df[clean_df['Is_Hetero'] == True]
        if not hetero_subset.empty:
            plot_task_distribution_chart(
                hetero_subset, 
                title="Task Distribution: Heterogeneous Tasks", 
                filename="distrib_tasks_Hetero.png"
            )
    else:
        print("No CSV files found.")
