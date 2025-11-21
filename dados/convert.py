import json
import csv
import pandas as pd
import os
import glob
from datetime import datetime

def formatar_numero(valor):
    """
    Formata números para ter até 3 casas decimais
    """
    if isinstance(valor, (int, float)):
        return round(valor, 3)
    return valor

def calcular_estatisticas(episode_data):
    """
    Calcula estatísticas especiais dos dados
    """
    episodios = episode_data['episodes']
    distancias = episode_data['distances']
    tempos = episode_data['times']
    passos = episode_data['steps']
    
    estatisticas = {
        'primeiro_episodio_9m': None,
        'soma_passos_ate_9m': 0,
        'episodio_mais_rapido_9m': None,
        'tempo_minimo_9m': float('inf')
    }
    
    # Encontrar primeiro episódio que ultrapassou 9m e soma de passos até ele
    passos_acumulados = 0
    encontrou_9m = False
    
    for i, distancia in enumerate(distancias):
        passos_acumulados += passos[i]
        
        if not encontrou_9m and distancia > 9.0:
            estatisticas['primeiro_episodio_9m'] = episodios[i]
            estatisticas['soma_passos_ate_9m'] = passos_acumulados
            encontrou_9m = True
        
        # Encontrar episódio que levou menos tempo para completar 9m
        if distancia > 9.0 and tempos[i] < estatisticas['tempo_minimo_9m']:
            estatisticas['episodio_mais_rapido_9m'] = episodios[i]
            estatisticas['tempo_minimo_9m'] = tempos[i]
    
    return estatisticas

def json_to_csv(json_file_path, csv_file_path):
    """
    Converte um arquivo JSON de dados de treinamento para CSV
    """
    try:
        # Ler o arquivo JSON
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extrair dados dos episódios
        episode_data = data['episode_data']
        episodes = episode_data['episodes']
        
        # Calcular estatísticas
        estatisticas = calcular_estatisticas(episode_data)
        
        # Criar lista de linhas para o CSV
        rows = []
        
        for i in range(len(episodes)):
            row = {
                'episode': episodes[i],
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
        
        # Escrever para CSV
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Escrever cabeçalho de informações da sessão
            csvfile.write("# RESUMO ESTATÍSTICO DO TREINAMENTO\n")
            csvfile.write(f"# Arquivo: {os.path.basename(json_file_path)}\n")
            csvfile.write(f"# Data de conversão: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            csvfile.write("#" * 50 + "\n")
            csvfile.write(f"# Robot: {data['session_info']['robot']}\n")
            csvfile.write(f"# Algorithm: {data['session_info']['algorithm']}\n")
            csvfile.write(f"# Environment: {data['session_info']['environment']}\n")
            csvfile.write(f"# Total Steps, {data['session_info']['total_steps']}\n")
            csvfile.write(f"# Total Episodes, {data['session_info']['total_episodes']}\n")
            csvfile.write(f"# Best Reward, {formatar_numero(data['tracker_status']['best_reward'])}\n")
            csvfile.write(f"# Best Distance, {formatar_numero(data['tracker_status']['best_distance'])}\n")
            csvfile.write("#" * 50 + "\n")
            csvfile.write("# ESTATÍSTICAS DE DESEMPENHO\n")
            
            if estatisticas['primeiro_episodio_9m']:
                csvfile.write(f"# Primeiro episódio >9m, {estatisticas['primeiro_episodio_9m']}\n")
                csvfile.write(f"# Soma de passos até >9m, {estatisticas['soma_passos_ate_9m']}\n")
            else:
                csvfile.write("# Primeiro episódio >9m, Nenhum episódio atingiu 9m\n")
                csvfile.write("# Soma de passos até >9m, N/A\n")
            
            if estatisticas['episodio_mais_rapido_9m']:
                csvfile.write(f"# Episódio mais rápido para >9m, {estatisticas['episodio_mais_rapido_9m']}\n")
                csvfile.write(f"# Tempo mínimo para >9m, {formatar_numero(estatisticas['tempo_minimo_9m'])}s\n")
            else:
                csvfile.write("# Episódio mais rápido para >9m, Nenhum episódio atingiu 9m\n")
                csvfile.write("# Tempo mínimo para >9m, N/A\n")
            
            # Estatísticas adicionais
            recompensa_maxima = max(episode_data['rewards'])
            distancia_minima = min(episode_data['distances'])
            csvfile.write(f"# Maior recompensa, {formatar_numero(recompensa_maxima)}\n")
            csvfile.write(f"# Menor distância, {formatar_numero(distancia_minima)}\n")
            csvfile.write("#" * 50 + "\n")
            
            # Escrever dados
            if rows:
                fieldnames = rows[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        # Mostrar estatísticas no console
        print(f"✓ Convertido: {os.path.basename(json_file_path)} -> {os.path.basename(csv_file_path)}")
        print(f"  Episódios: {len(rows)}, Métricas: {len(rows[0]) if rows else 0}")
        print(f"  Estatísticas:")
        if estatisticas['primeiro_episodio_9m']:
            print(f"    - Primeiro >9m: Ep.{estatisticas['primeiro_episodio_9m']} (Passos: {estatisticas['soma_passos_ate_9m']})")
            print(f"    - Mais rápido >9m: Ep.{estatisticas['episodio_mais_rapido_9m']} ({estatisticas['tempo_minimo_9m']:.1f}s)")
        else:
            print(f"    - Nenhum episódio atingiu 9m")
        print(f"    - Maior recompensa: {recompensa_maxima:.1f}")
        print(f"    - Menor distância: {distancia_minima:.3f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro ao converter {json_file_path}: {str(e)}")
        return False

def json_to_csv_pandas(json_file_path, csv_file_path):
    """
    Versão alternativa usando pandas (mais simples)
    """
    try:
        # Ler o arquivo JSON
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Calcular estatísticas
        estatisticas = calcular_estatisticas(data['episode_data'])
        
        # Criar DataFrame a partir dos dados dos episódios
        episode_data = data['episode_data']
        
        # Converter para DataFrame
        df = pd.DataFrame()
        
        # Adicionar todas as colunas do episode_data
        for key, values in episode_data.items():
            if key not in ['success']:  # Não formatar booleanos
                # Aplicar formatação de 3 casas decimais para colunas numéricas
                if isinstance(values[0], (int, float)):
                    df[key] = [formatar_numero(v) for v in values]
                else:
                    df[key] = values
            else:
                df[key] = values
        
        # Criar arquivo CSV com cabeçalho informativo
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
            # Escrever cabeçalho informativo
            f.write("# RESUMO ESTATÍSTICO DO TREINAMENTO\n")
            f.write(f"# Arquivo: {os.path.basename(json_file_path)}\n")
            f.write(f"# Data de conversão: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("#" * 50 + "\n")
            f.write(f"# Robot: {data['session_info']['robot']}\n")
            f.write(f"# Algorithm: {data['session_info']['algorithm']}\n")
            f.write(f"# Total Episodes, {data['session_info']['total_episodes']}\n")
            
            if estatisticas['primeiro_episodio_9m']:
                f.write(f"# Primeiro episódio >9m, {estatisticas['primeiro_episodio_9m']}\n")
                f.write(f"# Soma de passos até >9m, {estatisticas['soma_passos_ate_9m']}\n")
                f.write(f"# Episódio mais rápido >9m, {estatisticas['episodio_mais_rapido_9m']}\n")
            
            f.write("#" * 50 + "\n")
        
        # Adicionar dados ao CSV
        df.to_csv(csv_file_path, mode='a', index=False)
        
        print(f"✓ Convertido (pandas): {os.path.basename(json_file_path)} -> {os.path.basename(csv_file_path)}")
        print(f"  Shape: {df.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erro ao converter {json_file_path} com pandas: {str(e)}")
        return False

def converter_todos_arquivos_json(pasta='.', usar_pandas=False):
    """
    Converte todos os arquivos JSON na pasta para CSV
    """
    # Encontrar todos os arquivos JSON
    padrao_json = os.path.join(pasta, '*.json')
    arquivos_json = glob.glob(padrao_json)
    
    if not arquivos_json:
        print("Nenhum arquivo JSON encontrado na pasta atual.")
        return
    
    print(f"Encontrados {len(arquivos_json)} arquivo(s) JSON:")
    
    sucessos = 0
    for arquivo_json in arquivos_json:
        # Gerar nome do arquivo CSV
        nome_base = os.path.splitext(arquivo_json)[0]
        arquivo_csv = nome_base + '.csv'
        
        # Converter arquivo
        if usar_pandas:
            sucesso = json_to_csv_pandas(arquivo_json, arquivo_csv)
        else:
            sucesso = json_to_csv(arquivo_json, arquivo_csv)
        
        if sucesso:
            sucessos += 1
        print()  # Linha em branco entre arquivos
    
    print(f"Resumo: {sucessos}/{len(arquivos_json)} arquivos convertidos com sucesso!")

def converter_arquivo_especifico(arquivo_json, usar_pandas=False):
    """
    Converte um arquivo JSON específico para CSV
    """
    if not os.path.exists(arquivo_json):
        print(f"Arquivo não encontrado: {arquivo_json}")
        return
    
    # Gerar nome do arquivo CSV
    nome_base = os.path.splitext(arquivo_json)[0]
    arquivo_csv = nome_base + '.csv'
    
    # Converter arquivo
    if usar_pandas:
        json_to_csv_pandas(arquivo_json, arquivo_csv)
    else:
        json_to_csv(arquivo_json, arquivo_csv)

# Uso do script
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Converter arquivos JSON de treinamento para CSV')
    parser.add_argument('--arquivo', '-a', type=str, help='Converter um arquivo específico')
    parser.add_argument('--pasta', '-p', type=str, default='.', help='Pasta para procurar arquivos JSON')
    parser.add_argument('--pandas', action='store_true', help='Usar método pandas para conversão')
    parser.add_argument('--todos', '-t', action='store_true', help='Converter todos os arquivos JSON da pasta')
    
    args = parser.parse_args()
    
    if args.todos:
        # Converter todos os arquivos JSON da pasta
        converter_todos_arquivos_json(args.pasta, args.pandas)
    elif args.arquivo:
        # Converter um arquivo específico
        converter_arquivo_especifico(args.arquivo, args.pandas)
    else:
        # Modo interativo
        print("=== Conversor JSON para CSV ===")
        print("Características:")
        print("- Todos os números com 3 casas decimais")
        print("- Resumo estatístico incluído no CSV")
        print("- Estatísticas de desempenho (9m)")
        print()
        
        opcao = input("Converter (1) um arquivo específico ou (2) todos os JSON da pasta? [1/2]: ")
        
        if opcao == "1":
            arquivo = input("Nome do arquivo JSON: ")
            usar_pandas = input("Usar pandas? [s/N]: ").lower().startswith('s')
            converter_arquivo_especifico(arquivo, usar_pandas)
        else:
            pasta = input("Pasta (enter para pasta atual): ") or "."
            usar_pandas = input("Usar pandas? [s/N]: ").lower().startswith('s')
            converter_todos_arquivos_json(pasta, usar_pandas)