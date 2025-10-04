# metrics_saver.py
import csv
import math
import os
import json
import numpy as np
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


def calculate_energy_metrics(joint_torques_history, joint_velocities_history, timestep):
    """
    Calcula métricas de energia baseadas no histórico completo
    Custo Energético = Σ|torque * velocity| * Δt por episódio
    """
    if not joint_torques_history or not joint_velocities_history:
        return {"total_energy": 0, "avg_power": 0, "energy_efficiency": 0}

    total_energy = 0
    power_values = []

    for torques, velocities in zip(joint_torques_history, joint_velocities_history):
        # Potência instantânea = Σ|torque * velocity|
        instant_power = sum(abs(t * v) for t, v in zip(torques, velocities))
        energy_contribution = instant_power * timestep
        total_energy += energy_contribution
        power_values.append(instant_power)

    avg_power = np.mean(power_values) if power_values else 0
    energy_efficiency = 1 / (total_energy + 1e-6)  # Eficiência = 1/energia (quanto maior, melhor)

    return {
        "total_energy": total_energy,
        "avg_power": avg_power,
        "energy_efficiency": energy_efficiency,
        "max_power": max(power_values) if power_values else 0,
        "min_power": min(power_values) if power_values else 0,
    }


def calculate_stability_metrics(orientation_history):
    """
    Estabilidade baseada na variância da inclinação
    orientation_history: lista de tuples (roll, pitch, yaw)
    """
    if not orientation_history:
        return {"stability_roll": 0, "stability_pitch": 0, "stability_yaw": 0, "overall_stability": 0, "max_roll": 0, "max_pitch": 0}

    rolls, pitches, yaws = zip(*orientation_history)

    stability_metrics = {
        "stability_roll": np.std(rolls),
        "stability_pitch": np.std(pitches),
        "stability_yaw": np.std(yaws),
        "overall_stability": (np.std(rolls) + np.std(pitches) + np.std(yaws)) / 3,
        "max_roll": max(abs(r) for r in rolls),
        "max_pitch": max(abs(p) for p in pitches),
        "mean_roll": np.mean(rolls),
        "mean_pitch": np.mean(pitches),
    }

    return stability_metrics


def calculate_gait_metrics(foot_contact_history, step_timesteps):
    """
    Métricas de marcha e padrão de caminhada
    """
    if not foot_contact_history:
        return {"step_regularity": 0, "stance_phase_ratio": 0, "double_support_ratio": 0}

    # Calcular regularidade dos passos
    step_durations = []
    current_step_duration = 0

    for contact in foot_contact_history:
        current_step_duration += 1
        if contact:  # Contato detectado
            step_durations.append(current_step_duration)
            current_step_duration = 0

    step_regularity = 1 - (np.std(step_durations) / np.mean(step_durations)) if step_durations else 0

    # Calcular fases da marcha
    total_stance = sum(1 for contact in foot_contact_history if contact)
    total_swing = len(foot_contact_history) - total_stance
    stance_phase_ratio = total_stance / len(foot_contact_history) if foot_contact_history else 0

    return {
        "step_regularity": max(0, step_regularity),  # Evitar valores negativos
        "stance_phase_ratio": stance_phase_ratio,
        "swing_phase_ratio": 1 - stance_phase_ratio,
        "step_count": len(step_durations),
        "avg_step_duration": np.mean(step_durations) if step_durations else 0,
    }


def calculate_cross_metrics(origin_metrics, target_metrics):
    """
    Métricas de transferência cruzada (RF-09)
    """
    if origin_metrics["avg_time"] == 0:  # Evitar divisão por zero
        ΔTm = 0
    else:
        ΔTm = (target_metrics["avg_time"] - origin_metrics["avg_time"]) / origin_metrics["avg_time"]

    ΔSuccess = target_metrics["success_rate"] - origin_metrics["success_rate"]

    # Eficiência da transferência (quanto mais próximo de 1, melhor)
    transfer_efficiency = 1 - abs(ΔTm)

    # Penalidade por queda na taxa de sucesso
    success_penalty = max(0, -ΔSuccess * 2)  # Penalidade dobrada por perda de sucesso

    overall_transfer_score = transfer_efficiency - success_penalty

    return {
        "ΔTm": ΔTm,  # Variação relativa de tempo
        "ΔSuccess": ΔSuccess,
        "transfer_efficiency": transfer_efficiency,
        "success_penalty": success_penalty,
        "overall_transfer_score": overall_transfer_score,
        "transfer_type": "positive" if ΔTm < 0 else "negative",  # Negativo = melhor performance
    }


def calculate_confidence_intervals(times, confidence=0.95):
    """
    Calcula intervalo de confiança sem scipy.stats
    Usa aproximação normal para amostras grandes, t-student para pequenas
    """
    if not times or len(times) < 2:
        return {"lower": 0, "upper": 0, "mean": 0, "margin_error": 0}

    mean_time = np.mean(times)
    std_time = np.std(times)
    n = len(times)

    # Valores críticos para intervalos de confiança comum
    # Para 95% de confiança
    z_value_95 = 1.96  # Valor z para 95%
    t_values_95 = {  # Valores t para 95% (aproximados)
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        15: 2.145,
        20: 2.093,
        30: 2.042,
        50: 2.009,
        100: 1.984,
    }

    if n >= 30:
        # Usar distribuição normal para amostras grandes
        critical_value = z_value_95
    else:
        # Usar distribuição t para amostras pequenas
        critical_value = t_values_95.get(n, 2.0)  # Fallback para 2.0

    margin_error = critical_value * (std_time / math.sqrt(n))

    return {
        "lower": max(0, mean_time - margin_error),  # Não permitir tempo negativo
        "upper": mean_time + margin_error,
        "mean": mean_time,
        "margin_error": margin_error,
        "confidence_level": confidence,
        "critical_value": critical_value,
    }


def calculate_simple_confidence(times, sigma_multiplier=2):
    """
    Versão simplificada usando múltiplos do desvio padrão
    sigma_multiplier=2 ≈ 95% confiança para distribuição normal
    """
    if not times:
        return {"lower": 0, "upper": 0, "mean": 0}

    mean_time = np.mean(times)
    std_time = np.std(times) if len(times) > 1 else 0

    margin_error = sigma_multiplier * std_time

    return {"lower": max(0, mean_time - margin_error), "upper": mean_time + margin_error, "mean": mean_time, "margin_error": margin_error}


def compile_comprehensive_metrics(episode_data, joint_data, orientation_data, foot_contact_data):
    """
    Compila todas as métricas em um relatório completo
    """
    # Métricas básicas de tempo
    time_metrics = {
        "avg_time": np.mean(episode_data["times"]) if episode_data["times"] else 0,
        "std_time": np.std(episode_data["times"]) if len(episode_data["times"]) > 1 else 0,
        "success_rate": np.mean(episode_data["successes"]) if episode_data["successes"] else 0,
        "success_count": sum(episode_data["successes"]) if episode_data["successes"] else 0,
    }

    # Adicionar intervalo de confiança (usando versão simples)
    time_metrics.update(calculate_simple_confidence(episode_data["times"]))

    # Métricas de energia
    energy_metrics = calculate_energy_metrics(joint_data["torques"], joint_data["velocities"], joint_data.get("timestep", 0.02))

    # Métricas de estabilidade
    stability_metrics = calculate_stability_metrics(orientation_data["history"])

    # Métricas de marcha
    gait_metrics = calculate_gait_metrics(foot_contact_data["history"], foot_contact_data.get("timestep", 0.02))

    # Métrica composta de performance
    performance_score = (
        time_metrics["success_rate"] * 0.4
        + (1 - min(time_metrics["avg_time"] / 20, 1)) * 0.3  # Normalizar tempo (max 20s)
        + stability_metrics["overall_stability"] * 0.2
        + energy_metrics["energy_efficiency"] * 0.1
    )

    comprehensive_report = {
        "timestamp": datetime.now().isoformat(),
        "time_metrics": time_metrics,
        "energy_metrics": energy_metrics,
        "stability_metrics": stability_metrics,
        "gait_metrics": gait_metrics,
        "performance_score": performance_score,
        "total_episodes": len(episode_data["times"]),
    }

    return comprehensive_report
