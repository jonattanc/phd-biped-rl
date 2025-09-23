# metrics_saver.py
import csv
import os
import json
from datetime import datetime
import pandas as pd

def save_complexity_metrics(metrics, circuit_name, avatar_name, role="AE", seed=42, hyperparams=None, output_dir="logs/data"):
    """
    Salva as métricas de complexidade em um arquivo CSV.
    Formato completo conforme RF-13.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Nome do arquivo com timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"complexity_{circuit_name}_{avatar_name}_{role}_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    # Converter hyperparams para string JSON se for dicionário
    if hyperparams and isinstance(hyperparams, dict):
        hyperparams_str = json.dumps(hyperparams)
    else:
        hyperparams_str = str(hyperparams) if hyperparams else ""

    if "total_times" not in metrics or not metrics["total_times"]:
        print(f"[AVISO] Nenhum dado de episódio para salvar para {circuit_name}_{avatar_name}")
        return None, None
    
    # Salvar dados de cada episódio
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["timestamp", "circuit", "avatar", "role", "seed", "repetition", "time_total", "success", "hyperparams"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        for i, episode_time in enumerate(metrics["total_times"]):
            success = metrics.get("success_count", 0) > i if "success_count" in metrics else False
            writer.writerow(
                {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "circuit": circuit_name,
                    "avatar": avatar_name,
                    "role": role,
                    "seed": seed,
                    "repetition": i + 1,
                    "time_total": episode_time,
                    "success": success,
                    "hyperparams": hyperparams_str,
                }
            )

    # Salvar também um resumo das métricas
    summary_filename = f"summary_{circuit_name}_{avatar_name}_{role}_{timestamp}.csv"
    summary_path = os.path.join(output_dir, summary_filename)

    total_times = metrics.get("total_times", [])
    min_time = min(total_times) if total_times else 0
    max_time = max(total_times) if total_times else 0
    
    summary_data = {
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "circuit": [circuit_name],
        "avatar": [avatar_name],
        "role": [role],
        "seed": [seed],
        "avg_time": [metrics["avg_time"]],
        "std_time": [metrics["std_time"]],
        "success_rate": [metrics["success_rate"]],
        "num_episodes": [len(metrics["total_times"])],
        "min_time": [min_time],
        "max_time": [max_time],
        "hyperparams": [hyperparams_str],
    }

    pd.DataFrame(summary_data).to_csv(summary_path, index=False)

    print(f"[INFO] Métricas detalhadas salvas em: {filepath}")
    print(f"[INFO] Resumo salvo em: {summary_path}")
    return filepath, summary_path


def compile_results(pattern="*", output_file="logs/data/compiled_results.csv"):
    """
    Compila todos os resultados CSV baseados no padrão.
    """
    import glob

    csv_files = glob.glob(os.path.join("logs/data", f"{pattern}.csv"))

    if not csv_files:
        print(f"Nenhum arquivo encontrado com padrão: {pattern}")
        return None

    all_data = []

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            df["source_file"] = os.path.basename(csv_file)
            all_data.append(df)
        except Exception as e:
            print(f"Erro ao ler {csv_file}: {e}")

    if all_data:
        compiled_df = pd.concat(all_data, ignore_index=True)
        compiled_df.to_csv(output_file, index=False)
        print(f"Resultados compilados salvos em: {output_file}")
        return compiled_df

    return None


def generate_report(circuit_name=None, avatar_name=None):
    """
    Gera um relatório com base nos dados compilados.
    """
    compiled_df = compile_results()

    if compiled_df is None:
        print("Nenhum dado para gerar relatório")
        return

    # Filtrar por circuito e avatar se especificado
    if circuit_name:
        compiled_df = compiled_df[compiled_df["circuit"] == circuit_name]
    if avatar_name:
        compiled_df = compiled_df[compiled_df["avatar"] == avatar_name]

    if compiled_df.empty:
        print("Nenhum dado encontrado com os filtros especificados")
        return

    print("\n=== RELATÓRIO DE DESEMPENHO ===")

    # Agrupar por circuito e avatar
    grouped = compiled_df.groupby(["circuit", "avatar", "role"])

    for (circuit, avatar, role), group in grouped:
        print(f"\n--- {circuit} | {avatar} | {role} ---")
        print(f"Episódios: {len(group)}")
        print(f"Taxa de sucesso: {group['success'].mean() * 100:.1f}%")
        print(f"Tempo médio: {group['time_total'].mean():.2f}s ± {group['time_total'].std():.2f}s")
        print(f"Melhor tempo: {group['time_total'].min():.2f}s")
        print(f"Pior tempo: {group['time_total'].max():.2f}s")
