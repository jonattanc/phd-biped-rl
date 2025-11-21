import json
import csv
import os
import glob

def formatar_numero(valor):
    """Formata números para ter até 3 casas decimais"""
    if isinstance(valor, (int, float)):
        return round(valor, 3)
    return valor

def calcular_estatisticas(episode_data):
    """Calcula estatísticas dos dados"""
    episodios = episode_data['episodes']
    distancias = episode_data['distances']
    tempos = episode_data['times']
    passos = episode_data['steps']
    recompensas = episode_data['rewards']
    sucessos = episode_data['success']
    
    estatisticas = {
        'primeiro_episodio_9m': None,
        'soma_passos_ate_9m': 0,
        'episodio_mais_rapido_9m': None,
        'tempo_minimo_9m': float('inf'),
        'maior_recompensa': max(recompensas) if recompensas else 0,
        'episodio_maior_recompensa': episodios[recompensas.index(max(recompensas))] if recompensas else None,
        'menor_distancia': min(distancias) if distancias else 0,
        'episodio_menor_distancia': episodios[distancias.index(min(distancias))] if distancias else None,
        'media_recompensa': sum(recompensas) / len(recompensas) if recompensas else 0,
        'media_distancia': sum(distancias) / len(distancias) if distancias else 0,
        'total_sucessos': sum(sucessos),
        'taxa_sucesso': sum(sucessos) / len(sucessos) if sucessos else 0,
        'total_passos': sum(passos),
        'total_tempo': sum(tempos)
    }
    
    passos_acumulados = 0
    encontrou_9m = False
    
    for i, distancia in enumerate(distancias):
        passos_acumulados += passos[i]
        
        if not encontrou_9m and distancia > 9.0:
            estatisticas['primeiro_episodio_9m'] = episodios[i]
            estatisticas['soma_passos_ate_9m'] = passos_acumulados
            encontrou_9m = True
        
        if distancia > 9.0 and tempos[i] < estatisticas['tempo_minimo_9m']:
            estatisticas['episodio_mais_rapido_9m'] = episodios[i]
            estatisticas['tempo_minimo_9m'] = tempos[i]
    
    return estatisticas

def json_para_csv(json_file_path, csv_file_path):
    """Converte um arquivo JSON para CSV"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        episode_data = data['episode_data']
        estatisticas = calcular_estatisticas(episode_data)
        
        rows = []
        for i in range(len(episode_data['episodes'])):
            row = {
                'episode': episode_data['episodes'][i],
                'reward': formatar_numero(episode_data['rewards'][i]),
                'time': formatar_numero(episode_data['times'][i]),
                'distance': formatar_numero(episode_data['distances'][i]),
                'imu_x': formatar_numero(episode_data['imu_x'][i]),
                'imu_y': formatar_numero(episode_data['imu_y'][i]),
                'imu_z': formatar_numero(episode_data['imu_z'][i]),
                'roll_deg': formatar_numero(episode_data['roll_deg'][i]),
                'pitch_deg': formatar_numero(episode_data['pitch_deg'][i]),
                'yaw_deg': formatar_numero(episode_data['yaw_deg'][i]),
                'imu_average_x_vel': formatar_numero(episode_data['imu_average_x_vel'][i]),
                'imu_average_y_vel': formatar_numero(episode_data['imu_average_y_vel'][i]),
                'imu_average_z_vel': formatar_numero(episode_data['imu_average_z_vel'][i]),
                'roll_vel_deg': formatar_numero(episode_data['roll_vel_deg'][i]),
                'pitch_vel_deg': formatar_numero(episode_data['pitch_vel_deg'][i]),
                'yaw_vel_deg': formatar_numero(episode_data['yaw_vel_deg'][i]),
                'filtered_reward': formatar_numero(episode_data['filtered_rewards'][i]),
                'filtered_time': formatar_numero(episode_data['filtered_times'][i]),
                'filtered_distance': formatar_numero(episode_data['filtered_distances'][i]),
                'filtered_imu_x': formatar_numero(episode_data['filtered_imu_x'][i]),
                'filtered_imu_y': formatar_numero(episode_data['filtered_imu_y'][i]),
                'filtered_imu_z': formatar_numero(episode_data['filtered_imu_z'][i]),
                'filtered_roll_deg': formatar_numero(episode_data['filtered_roll_deg'][i]),
                'filtered_pitch_deg': formatar_numero(episode_data['filtered_pitch_deg'][i]),
                'filtered_yaw_deg': formatar_numero(episode_data['filtered_yaw_deg'][i]),
                'filtered_imu_average_x_vel': formatar_numero(episode_data['filtered_imu_average_x_vel'][i]),
                'filtered_imu_average_y_vel': formatar_numero(episode_data['filtered_imu_average_y_vel'][i]),
                'filtered_imu_average_z_vel': formatar_numero(episode_data['filtered_imu_average_z_vel'][i]),
                'filtered_roll_vel_deg': formatar_numero(episode_data['filtered_roll_vel_deg'][i]),
                'filtered_pitch_vel_deg': formatar_numero(episode_data['filtered_pitch_vel_deg'][i]),
                'filtered_yaw_vel_deg': formatar_numero(episode_data['filtered_yaw_vel_deg'][i]),
                'reward_filtered': formatar_numero(episode_data['rewards_filtered'][i]),
                'steps': episode_data['steps'][i],
                'success': episode_data['success'][i],
                'imu_x_vel': formatar_numero(episode_data['imu_x_vel'][i]),
                'imu_y_vel': formatar_numero(episode_data['imu_y_vel'][i]),
                'imu_z_vel': formatar_numero(episode_data['imu_z_vel'][i]),
                'roll': formatar_numero(episode_data['roll'][i]),
                'pitch': formatar_numero(episode_data['pitch'][i]),
                'yaw': formatar_numero(episode_data['yaw'][i]),
                'roll_vel': formatar_numero(episode_data['roll_vel'][i]),
                'pitch_vel': formatar_numero(episode_data['pitch_vel'][i]),
                'yaw_vel': formatar_numero(episode_data['yaw_vel'][i])
            }
            rows.append(row)
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Cabeçalho com estatísticas
            csvfile.write("# RESUMO ESTATÍSTICO - COMPARAÇÃO DE TREINAMENTOS\n")
            csvfile.write("##################################################\n")
            csvfile.write(f"# Arquivo: {os.path.basename(json_file_path)}\n")
            csvfile.write(f"# Robot: {data['session_info']['robot']}\n")
            csvfile.write(f"# Algorithm: {data['session_info']['algorithm']}\n")
            csvfile.write(f"# Environment: {data['session_info']['environment']}\n")
            csvfile.write(f"# Total Steps, {data['session_info']['total_steps']}\n")
            csvfile.write(f"# Total Episodes, {data['session_info']['total_episodes']}\n")
            csvfile.write(f"# Best Reward, {formatar_numero(data['tracker_status']['best_reward'])}\n")
            csvfile.write(f"# Best Distance, {formatar_numero(data['tracker_status']['best_distance'])}\n")
            csvfile.write("##################################################\n")
            csvfile.write("# MÉTRICAS DE DESEMPENHO\n")
            csvfile.write(f"# Maior recompensa, {formatar_numero(estatisticas['maior_recompensa'])} , {estatisticas['episodio_maior_recompensa']}\n")
            csvfile.write(f"# Menor distância, {formatar_numero(estatisticas['menor_distancia'])} , {estatisticas['episodio_menor_distancia']}\n")
            csvfile.write(f"# Média recompensa, {formatar_numero(estatisticas['media_recompensa'])}\n")
            csvfile.write(f"# Média distância, {formatar_numero(estatisticas['media_distancia'])}\n")
            csvfile.write(f"# Sucessos, {formatar_numero(estatisticas['taxa_sucesso']*100)}, {estatisticas['total_sucessos']}/{len(episode_data['success'])}\n")
            csvfile.write(f"# Total passos, {estatisticas['total_passos']}\n")
            csvfile.write(f"# Total tempo (s), {formatar_numero(estatisticas['total_tempo'])}\n")
            
            if estatisticas['primeiro_episodio_9m']:
                csvfile.write(f"# Primeiro episódio >9m, {estatisticas['primeiro_episodio_9m']}\n")
                csvfile.write(f"# Soma passos até >9m, {estatisticas['soma_passos_ate_9m']}\n")
                csvfile.write(f"# Episódio mais rápido >9m, {estatisticas['episodio_mais_rapido_9m']}\n")
                csvfile.write(f"# Tempo mínimo >9m (s), {formatar_numero(estatisticas['tempo_minimo_9m'])}\n")
            else:
                csvfile.write("# Nenhum episódio atingiu 9m\n")
            
            csvfile.write("##################################################\n")
            csvfile.write("# DADOS DOS EPISÓDIOS\n")
            
            # Dados dos episódios
            if rows:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        return True
        
    except Exception as e:
        print(f"Erro ao converter {json_file_path}: {str(e)}")
        return False

def converter_todos_json(pasta='.'):
    """Converte todos os arquivos JSON na pasta para CSV"""
    padrao_json = os.path.join(pasta, '*.json')
    arquivos_json = glob.glob(padrao_json)
    
    if not arquivos_json:
        print("Nenhum arquivo JSON encontrado.")
        return
    
    for arquivo_json in arquivos_json:
        nome_base = os.path.splitext(arquivo_json)[0]
        arquivo_csv = nome_base + '.csv'
        
        if json_para_csv(arquivo_json, arquivo_csv):
            print(f"Convertido: {os.path.basename(arquivo_json)} -> {os.path.basename(arquivo_csv)}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        arquivo_json = sys.argv[1]
        if os.path.exists(arquivo_json):
            nome_base = os.path.splitext(arquivo_json)[0]
            arquivo_csv = nome_base + '.csv'
            json_para_csv(arquivo_json, arquivo_csv)
        else:
            print(f"Arquivo não encontrado: {arquivo_json}")
    else:
        converter_todos_json()