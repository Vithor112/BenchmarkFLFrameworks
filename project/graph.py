import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

df = pd.read_csv('run_metrics.csv')


df['Adjusted_Window_min'] = df['Adjusted_Window_s'] / 60.0

df['Server_Net_Tx_KBps'] = df['Server_Net_Tx_Bps'] / 1024.0
df['Server_Net_Rx_KBps'] = df['Server_Net_Rx_Bps'] / 1024.0
df['Clients_Avg_Net_Tx_KBps'] = df['Clients_Avg_Net_Tx_Bps'] / 1024.0
df['Clients_Avg_Net_Rx_KBps'] = df['Clients_Avg_Net_Rx_Bps'] / 1024.0
# -----------------------------------------

sns.set_theme(style="whitegrid", palette="muted")
framework_palette = {"nvidiaFlare": "#76B900", "flower": "#FFB74D", "fedbiomed": "#4FC3F7"}

df_scenario_clients = df[(df['Param_Rounds'] == 3) & 
                   ((df['Param_Batch_Size'] == 32) | ((df['Param_Clients'] == 10) & (df['Param_Batch_Size'] == 16)))]

df_scenario_rounds = df[(df['Param_Clients'] == 3) & (df['Param_Batch_Size'] == 32)]

df_scenario_batch_sizes = df[(df['Param_Clients'] == 3) & (df['Param_Rounds'] == 3)]


def plot_metrics_scenario(data, x_var, x_label, title_prefix):
    metrics_to_plot = [
        ('Adjusted_Window_min', 'Tempo de Duração (min)'),
        ('Server_CPU_Cores', 'Núcleos de CPU do Servidor'),
        ('Server_Memory_Bytes', 'Memória do Servidor (Bytes)'),
        ('Clients_Avg_CPU_Cores', 'Média de Núcleos de CPU (Clientes)'),
        ('Clients_Avg_Memory_Bytes', 'Média de Memória (Clientes - Bytes)'),
        ('Clients_Avg_Net_Tx_KBps', 'Média Transmitida (Clientes - KBps)'),
        ('Clients_Avg_Net_Rx_KBps', 'Média Recebida (Clientes - KBps)'),
        ('Clients_Avg_GPU_Util', 'Média de Uso de GPU (Clientes - %)')
    ]
    
    print(f"\n{'='*80}")
    print(f"GERANDO GRÁFICOS E ESTATÍSTICAS: {title_prefix.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    for col, ylabel in metrics_to_plot:
        plt.figure(figsize=(10, 6)) 
        
        sns.barplot(data=data, x=x_var, y=col, hue='Framework', 
                    palette=framework_palette, errorbar='sd', capsize=0.1)
        
        plt.title("")
        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel(x_label, fontsize=12)
        plt.legend(title='Framework', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        clean_ylabel = ylabel.replace(' ', '_').replace('(', '').replace(')', '').replace('%', 'pct').replace('-', '').replace('__', '_')
        filename = f"Grafico_{title_prefix}_{clean_ylabel}.png"
        plt.savefig(filename, bbox_inches='tight', dpi=300) 
        plt.close()
        
        print(f"\nMétrica: {ylabel} ({col})")
        stats = data.groupby(['Framework', x_var])[col].agg(['mean', 'std']).reset_index()
        for index, row in stats.iterrows():
            fw = row['Framework']
            vary_val = row[x_var]
            mean_val = row['mean']
            std_val = row['std'] if pd.notna(row['std']) else 0.0 
            print(f"  Framework: {fw: <12} | {x_var}: {vary_val: <3} | Média: {mean_val: >15.4f} | Desvio Padrão: {std_val: >15.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6)) 
    
    sns.barplot(ax=axes[0], data=data, x=x_var, y='Server_Net_Tx_KBps', hue='Framework', 
                palette=framework_palette, errorbar='sd', capsize=0.1)
    axes[0].set_title('Servidor - Transmitido (KBps)', fontsize=14)
    axes[0].set_ylabel('KBps', fontsize=12)
    axes[0].set_xlabel(x_label, fontsize=12)
    axes[0].get_legend().remove() 
    
    sns.barplot(ax=axes[1], data=data, x=x_var, y='Server_Net_Rx_KBps', hue='Framework', 
                palette=framework_palette, errorbar='sd', capsize=0.1)
    axes[1].set_title('Servidor - Recebido (KBps)', fontsize=14)
    axes[1].set_ylabel('KBps', fontsize=12)
    axes[1].set_xlabel(x_label, fontsize=12)
    axes[1].legend(title='Framework', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    filename_net = f"Grafico_{title_prefix}_Servidor_Rede_Tx_Rx_Combinado.png"
    plt.savefig(filename_net, bbox_inches='tight', dpi=300)
    plt.close()

    for col, title in [('Server_Net_Tx_KBps', 'Servidor - Transmitido'), ('Server_Net_Rx_KBps', 'Servidor - Recebido')]:
        print(f"\nMétrica: {title} ({col})")
        stats = data.groupby(['Framework', x_var])[col].agg(['mean', 'std']).reset_index()
        for index, row in stats.iterrows():
            fw = row['Framework']
            vary_val = row[x_var]
            mean_val = row['mean']
            std_val = row['std'] if pd.notna(row['std']) else 0.0 
            print(f"  Framework: {fw: <12} | {x_var}: {vary_val: <3} | Média: {mean_val: >15.4f} | Desvio Padrão: {std_val: >15.4f}")


def expand_accuracies(data):
    rows = []
    for index, row in data.iterrows():
        try:
            acc_dict = json.loads(row['Accuracies_Per_Round'])
            for round_key, acc_val in acc_dict.items():
                round_num = int(round_key.replace("Round ", "").strip())
                new_row = row.copy()
                new_row['Round_Number'] = round_num
                new_row['Accuracy'] = float(acc_val)
                rows.append(new_row)
        except Exception as e:
            pass
    return pd.DataFrame(rows)

def print_accuracy_stats(data, vary_col, scenario_name):
    df_acc = expand_accuracies(data)
    
    if df_acc.empty:
        print(f"Sem dados de acurácia para {scenario_name}.")
        return
        
    stats = df_acc.groupby(['Framework', vary_col, 'Round_Number'])['Accuracy'].agg(['mean', 'std']).reset_index()
    
    print(f"\n{'='*80}")
    print(f"ESTATÍSTICAS DE ACURÁCIA: {scenario_name.upper().replace('_', ' ')}")
    print(f"{'='*80}")
    
    for index, row in stats.iterrows():
        fw = row['Framework']
        vary_val = row[vary_col]
        rnd = row['Round_Number']
        mean_acc = row['mean']
        std_acc = row['std'] if pd.notna(row['std']) else 0.0
        
        print(f"Framework: {fw: <12} | {vary_col}: {vary_val: <3} | Rodada: {rnd: <2} | Média: {mean_acc:.4f} | Desvio Padrão: {std_acc:.4f}")
    print("\n")


print("Iniciando o processamento dos cenários...")

plot_metrics_scenario(df_scenario_clients, 'Param_Clients', 'Número de Clientes', 'Cenario_1_Clientes')
plot_metrics_scenario(df_scenario_rounds, 'Param_Rounds', 'Número de Rodadas', 'Cenario_2_Rodadas')
plot_metrics_scenario(df_scenario_batch_sizes, 'Param_Batch_Size', 'Tamanho do Lote', 'Cenario_3_Tamanho_do_Lote')

print("\nCalculando estatísticas de precisão isoladas...")
print_accuracy_stats(df_scenario_clients, 'Param_Clients', 'Cenario_1_Clientes')
print_accuracy_stats(df_scenario_rounds, 'Param_Rounds', 'Cenario_2_Rodadas')
print_accuracy_stats(df_scenario_batch_sizes, 'Param_Batch_Size', 'Cenario_3_Tamanho_do_Lote')

print("Processamento concluído com sucesso!")