import json
import csv
import os
import glob
from datetime import datetime

def formatar_numero_string(valor):
    """Formata números como string com exatamente 3 casas decimais"""
    if isinstance(valor, (int, float)):
        return f"{valor:.3f}"
    return str(valor)

def formatar_numero_float(valor):
    """Formata números como float com 3 casas decimais (para cálculos)"""
    if isinstance(valor, (int, float)):
        return float(f"{valor:.3f}")
    return valor

def formatar_booleano(valor):
    """Converte True/False para 1/0"""
    if isinstance(valor, bool):
        return 1 if valor else 0
    return valor

def calcular_estatisticas_completas(episode_data, tracker_status):
    """Calcula estatísticas completas dos dados"""
    episodios = episode_data['episodes']
    distancias = episode_data['distances']
    tempos = episode_data['times']
    passos = episode_data['steps']
    recompensas = episode_data['rewards']
    sucessos = episode_data['success']
    
    # Converter booleanos para inteiros para cálculos
    sucessos_int = [1 if s else 0 for s in sucessos]
    
    # Encontrar primeiro sucesso (>9m)
    primeiro_episodio_sucesso = None
    soma_passos_ate_primeiro_sucesso = 0
    tempo_acumulado_ate_primeiro_sucesso = 0 
    
    # Encontrar episódio mais rápido >9m
    episodio_mais_rapido_9m = None
    tempo_minimo_9m = float('inf')
    
    # Encontrar último salvamento (melhor recompensa)
    episodio_ultimo_salvamento = tracker_status.get('best_episode', len(episodios))
    melhor_recompensa = tracker_status.get('best_reward', max(recompensas) if recompensas else 0)
    
    passos_acumulados = 0
    tempo_acumulado = 0  
    encontrou_primeiro_sucesso = False
    passos_ate_ultimo_salvamento = 0
    tempo_ate_ultimo_salvamento = 0  
    
    for i, (distancia, sucesso, tempo_episodio) in enumerate(zip(distancias, sucessos_int, tempos)):
        passos_acumulados += passos[i]
        tempo_acumulado += tempo_episodio  
        
        # Primeiro sucesso
        if not encontrou_primeiro_sucesso and sucesso == 1:
            primeiro_episodio_sucesso = episodios[i]
            soma_passos_ate_primeiro_sucesso = passos_acumulados
            tempo_acumulado_ate_primeiro_sucesso = tempo_acumulado  
            encontrou_primeiro_sucesso = True
        
        # Episódio mais rápido >9m
        if sucesso == 1 and tempos[i] < tempo_minimo_9m:
            episodio_mais_rapido_9m = episodios[i]
            tempo_minimo_9m = tempos[i]
        
        # Passos e tempo até último salvamento
        if episodios[i] <= episodio_ultimo_salvamento:
            passos_ate_ultimo_salvamento = passos_acumulados
            tempo_ate_ultimo_salvamento = tempo_acumulado
    
    # Calcular sucessos após primeiro sucesso
    if primeiro_episodio_sucesso:
        indice_primeiro_sucesso = episodios.index(primeiro_episodio_sucesso)
        sucessos_apos_primeiro = sum(sucessos_int[indice_primeiro_sucesso:])
        total_episodios_apos_primeiro = len(episodios) - indice_primeiro_sucesso
        percentual_sucessos_apos_primeiro = sucessos_apos_primeiro / total_episodios_apos_primeiro if total_episodios_apos_primeiro > 0 else 0
    else:
        sucessos_apos_primeiro = 0
        percentual_sucessos_apos_primeiro = 0
    
    # Passos de aprendizagem residual
    if primeiro_episodio_sucesso:
        passos_residual = passos_ate_ultimo_salvamento - soma_passos_ate_primeiro_sucesso
        tempo_residual = tempo_ate_ultimo_salvamento - tempo_acumulado_ate_primeiro_sucesso  # NOVO: tempo residual
    else:
        passos_residual = 0
        tempo_residual = 0
    
    estatisticas = {
        # Métricas básicas
        'total_episodios': len(episodios),
        'melhor_distancia': formatar_numero_float(tracker_status.get('best_distance', max(distancias) if distancias else 0)),
        'maior_recompensa': formatar_numero_float(melhor_recompensa),
        'menor_distancia': formatar_numero_float(min(distancias) if distancias else 0),
        'media_recompensa': formatar_numero_float(sum(recompensas) / len(recompensas)) if recompensas else 0,
        'media_distancia': formatar_numero_float(sum(distancias) / len(distancias)) if distancias else 0,
        
        # Sucessos
        'episodios_sucesso': sum(sucessos_int),
        'percentual_sucessos': formatar_numero_float(sum(sucessos_int) / len(sucessos_int)) if sucessos_int else 0,
        'total_passos': sum(passos),
        'total_tempo': formatar_numero_float(sum(tempos)),
        
        # Primeiro sucesso
        'primeiro_episodio_sucesso': primeiro_episodio_sucesso,
        'soma_passos_ate_primeiro_sucesso': soma_passos_ate_primeiro_sucesso,
        'tempo_primeiro_sucesso_episodio': formatar_numero_float(tempos[episodios.index(primeiro_episodio_sucesso)]) if primeiro_episodio_sucesso else None,  # Tempo do episódio
        'tempo_treinamento_ate_primeiro_sucesso': formatar_numero_float(tempo_acumulado_ate_primeiro_sucesso) if primeiro_episodio_sucesso else None,  # Tempo TOTAL de treinamento
        
        # Performance >9m
        'episodio_mais_rapido_9m': episodio_mais_rapido_9m,
        'melhor_tempo_9m': formatar_numero_float(tempo_minimo_9m) if tempo_minimo_9m != float('inf') else None,
        
        # Evolução do aprendizado
        'percentual_sucessos_apos_primeiro': formatar_numero_float(percentual_sucessos_apos_primeiro),
        'passos_ate_ultimo_salvamento': passos_ate_ultimo_salvamento,
        'tempo_ate_ultimo_salvamento': formatar_numero_float(tempo_ate_ultimo_salvamento),
        'passos_aprendizagem_residual': passos_residual,
        'tempo_aprendizagem_residual': formatar_numero_float(tempo_residual),  
        
        # Informações do tracker
        'episodio_ultimo_salvamento': episodio_ultimo_salvamento,
        'steps_since_improvement': tracker_status.get('steps_since_improvement', 0),
        'auto_save_count': tracker_status.get('auto_save_count', 0)
    }
    
    return estatisticas

def json_para_csv(json_file_path, csv_file_path):
    """Converte um arquivo JSON para CSV"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        episode_data = data['episode_data']
        tracker_status = data['tracker_status']
        session_info = data['session_info']
        
        estatisticas = calcular_estatisticas_completas(episode_data, tracker_status)
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Cabeçalho com estatísticas COMPLETAS
            csvfile.write("# RESUMO ESTATÍSTICO - COMPARAÇÃO DE TREINAMENTOS\n")
            csvfile.write("##################################################\n")
            csvfile.write(f"# Arquivo: {os.path.basename(json_file_path)}\n")
            csvfile.write(f"# Robot: {session_info['robot']}\n")
            csvfile.write(f"# Algorithm: {session_info['algorithm']}\n")
            csvfile.write(f"# Environment: {session_info['environment']}\n")
            csvfile.write(f"# Data: {session_info.get('save_time', datetime.now().isoformat())}\n")
            csvfile.write("##################################################\n")
            csvfile.write("# MÉTRICAS DE DESEMPENHO\n")
            
            # Escrever todas as métricas no formato desejado
            metricas = [
                ('# Total Episódios', estatisticas['total_episodios']),
                ('# Melhor Distancia', estatisticas['melhor_distancia']),
                ('# Maior Recompensa', estatisticas['maior_recompensa']),
                ('# Menor Distância', estatisticas['menor_distancia']),
                ('# Média recompensa', estatisticas['media_recompensa']),
                ('# Média Distância', estatisticas['media_distancia']),
                ('# Episodios de sucesso', estatisticas['episodios_sucesso']),
                ('# Percentual de Sucessos', f"{estatisticas['percentual_sucessos']*100:.1f}%"),
                ('# Total de Passos', estatisticas['total_passos']),
                ('# Total tempo (s)', estatisticas['total_tempo']),
                ('# Primeiro episódio de Sucesso', estatisticas['primeiro_episodio_sucesso'] or 'Nenhum'),
                ('# Soma passos até Primeiro Sucesso', estatisticas['soma_passos_ate_primeiro_sucesso']),
                ('# Tempo do Primeiro Sucesso (s)', estatisticas['tempo_primeiro_sucesso'] or 'N/A'),
                ('# Número do Episódio mais rápido >9m', estatisticas['episodio_mais_rapido_9m'] or 'Nenhum'),
                ('# Melhor Tempo do Percurso (9m) (s)', estatisticas['melhor_tempo_9m'] or 'N/A'),
                ('# % Sucessos depois do primeiro sucesso', f"{estatisticas['percentual_sucessos_apos_primeiro']*100:.1f}%"),
                ('# Total de Passos até parar de evoluir', estatisticas['passos_ate_ultimo_salvamento']),
                ('# Tempo até >9m', estatisticas['tempo_ate_9m'] or 'N/A'),
                ('# Passos de Aprendizagem residual', estatisticas['passos_aprendizagem_residual']),
                ('# Episódio do último salvamento', estatisticas['episodio_ultimo_salvamento']),
                ('# Steps sem melhoria', estatisticas['steps_since_improvement']),
                ('# Número de auto-saves', estatisticas['auto_save_count'])
            ]
            
            for nome, valor in metricas:
                csvfile.write(f"{nome}, {valor}\n")
            
            csvfile.write("##################################################\n")
            csvfile.write("# DADOS DOS EPISÓDIOS\n")
            
            # Escrever cabeçalho dos dados
            headers = [
                'Pista', 'episodio', 'distancia', 'tempo', 'recompensa', 'passos', 'successos',
                'imu_x', 'imu_y', 'imu_z', 'roll_deg', 'pitch_deg', 'yaw_deg',
                'imu_average_x_vel', 'imu_average_y_vel', 'imu_average_z_vel',
                'roll_vel_deg', 'pitch_vel_deg', 'yaw_vel_deg',
                'imu_x_vel', 'imu_y_vel', 'imu_z_vel',
                'roll', 'pitch', 'yaw', 'roll_vel', 'pitch_vel', 'yaw_vel'
            ]
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            
            # Escrever dados dos episódios
            for i in range(len(episode_data['episodes'])):
                row = [
                    episode_data['episode_environments'][i],
                    episode_data['episodes'][i],
                    formatar_numero_string(episode_data['distances'][i]),
                    formatar_numero_string(episode_data['times'][i]),
                    formatar_numero_string(episode_data['rewards'][i]),
                    episode_data['steps'][i],
                    formatar_booleano(episode_data['success'][i]),
                    formatar_numero_string(episode_data['imu_x'][i]),
                    formatar_numero_string(episode_data['imu_y'][i]),
                    formatar_numero_string(episode_data['imu_z'][i]),
                    formatar_numero_string(episode_data['roll_deg'][i]),
                    formatar_numero_string(episode_data['pitch_deg'][i]),
                    formatar_numero_string(episode_data['yaw_deg'][i]),
                    formatar_numero_string(episode_data['imu_average_x_vel'][i]),
                    formatar_numero_string(episode_data['imu_average_y_vel'][i]),
                    formatar_numero_string(episode_data['imu_average_z_vel'][i]),
                    formatar_numero_string(episode_data['roll_vel_deg'][i]),
                    formatar_numero_string(episode_data['pitch_vel_deg'][i]),
                    formatar_numero_string(episode_data['yaw_vel_deg'][i]),
                    formatar_numero_string(episode_data['imu_x_vel'][i]),
                    formatar_numero_string(episode_data['imu_y_vel'][i]),
                    formatar_numero_string(episode_data['imu_z_vel'][i]),
                    formatar_numero_string(episode_data['roll'][i]),
                    formatar_numero_string(episode_data['pitch'][i]),
                    formatar_numero_string(episode_data['yaw'][i]),
                    formatar_numero_string(episode_data['roll_vel'][i]),
                    formatar_numero_string(episode_data['pitch_vel'][i]),
                    formatar_numero_string(episode_data['yaw_vel'][i])
                ]
                writer.writerow(row)
        
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