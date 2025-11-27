import json
import csv
import os
import glob
from datetime import datetime
from collections import defaultdict

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

def calcular_estatisticas_por_pista(episode_data, tracker_status):
    """Calcula estatísticas agrupadas por tipo de pista"""
    episodios = episode_data['episodes']
    distancias = episode_data['distances']
    tempos = episode_data['times']
    passos = episode_data['steps']
    recompensas = episode_data['rewards']
    sucessos = episode_data['success']
    pistas = episode_data['episode_environments']
    
    # Agrupar dados por pista
    dados_por_pista = defaultdict(lambda: {
        'episodios': [],
        'distancias': [],
        'tempos': [],
        'passos': [],
        'recompensas': [],
        'sucessos': [],
        'sucessos_int': []
    })
    
    for i, pista in enumerate(pistas):
        dados_por_pista[pista]['episodios'].append(episodios[i])
        dados_por_pista[pista]['distancias'].append(distancias[i])
        dados_por_pista[pista]['tempos'].append(tempos[i])
        dados_por_pista[pista]['passos'].append(passos[i])
        dados_por_pista[pista]['recompensas'].append(recompensas[i])
        dados_por_pista[pista]['sucessos'].append(sucessos[i])
        dados_por_pista[pista]['sucessos_int'].append(1 if sucessos[i] else 0)
    
    # Calcular estatísticas para cada pista
    estatisticas_por_pista = {}
    
    for pista, dados in dados_por_pista.items():
        if not dados['episodios']:
            continue
            
        # Encontrar primeiro sucesso para esta pista
        primeiro_episodio_sucesso_pista = None
        soma_passos_ate_primeiro_sucesso_pista = 0
        tempo_acumulado_ate_primeiro_sucesso_pista = 0
        
        passos_acumulados_pista = 0
        tempo_acumulado_pista = 0
        encontrou_primeiro_sucesso_pista = False
        
        for i, (sucesso, passos_ep, tempo_ep) in enumerate(zip(
            dados['sucessos_int'], dados['passos'], dados['tempos']
        )):
            passos_acumulados_pista += passos_ep
            tempo_acumulado_pista += tempo_ep
            
            if not encontrou_primeiro_sucesso_pista and sucesso == 1:
                primeiro_episodio_sucesso_pista = dados['episodios'][i]
                soma_passos_ate_primeiro_sucesso_pista = passos_acumulados_pista
                tempo_acumulado_ate_primeiro_sucesso_pista = tempo_acumulado_pista
                encontrou_primeiro_sucesso_pista = True
        
        # Encontrar episódio mais rápido >9m para esta pista
        episodio_mais_rapido_9m_pista = None
        tempo_minimo_9m_pista = float('inf')
        
        for i, (sucesso, tempo) in enumerate(zip(dados['sucessos_int'], dados['tempos'])):
            if sucesso == 1 and tempo < tempo_minimo_9m_pista:
                episodio_mais_rapido_9m_pista = dados['episodios'][i]
                tempo_minimo_9m_pista = tempo
        
        # Calcular sucessos após primeiro sucesso para esta pista
        if primeiro_episodio_sucesso_pista:
            indice_primeiro_sucesso = dados['episodios'].index(primeiro_episodio_sucesso_pista)
            sucessos_apos_primeiro = sum(dados['sucessos_int'][indice_primeiro_sucesso:])
            total_episodios_apos_primeiro = len(dados['episodios']) - indice_primeiro_sucesso
            percentual_sucessos_apos_primeiro = sucessos_apos_primeiro / total_episodios_apos_primeiro if total_episodios_apos_primeiro > 0 else 0
        else:
            sucessos_apos_primeiro = 0
            percentual_sucessos_apos_primeiro = 0
        
        estatisticas_por_pista[pista] = {
            'total_episodios': len(dados['episodios']),
            'melhor_distancia': formatar_numero_float(max(dados['distancias'])) if dados['distancias'] else 0,
            'maior_recompensa': formatar_numero_float(max(dados['recompensas'])) if dados['recompensas'] else 0,
            'menor_distancia': formatar_numero_float(min(dados['distancias'])) if dados['distancias'] else 0,
            'media_recompensa': formatar_numero_float(sum(dados['recompensas']) / len(dados['recompensas'])) if dados['recompensas'] else 0,
            'media_distancia': formatar_numero_float(sum(dados['distancias']) / len(dados['distancias'])) if dados['distancias'] else 0,
            'episodios_sucesso': sum(dados['sucessos_int']),
            'percentual_sucessos': formatar_numero_float(sum(dados['sucessos_int']) / len(dados['sucessos_int'])) if dados['sucessos_int'] else 0,
            'total_passos': sum(dados['passos']),
            'total_tempo': formatar_numero_float(sum(dados['tempos'])),
            'primeiro_episodio_sucesso': primeiro_episodio_sucesso_pista,
            'soma_passos_ate_primeiro_sucesso': soma_passos_ate_primeiro_sucesso_pista,
            'tempo_primeiro_sucesso_episodio': formatar_numero_float(dados['tempos'][dados['episodios'].index(primeiro_episodio_sucesso_pista)]) if primeiro_episodio_sucesso_pista else None,
            'tempo_treinamento_ate_primeiro_sucesso': formatar_numero_float(tempo_acumulado_ate_primeiro_sucesso_pista) if primeiro_episodio_sucesso_pista else None,
            'episodio_mais_rapido_9m': episodio_mais_rapido_9m_pista,
            'melhor_tempo_9m': formatar_numero_float(tempo_minimo_9m_pista) if tempo_minimo_9m_pista != float('inf') else None,
            'percentual_sucessos_apos_primeiro': formatar_numero_float(percentual_sucessos_apos_primeiro),
        }
    
    return estatisticas_por_pista

def calcular_estatisticas_gerais(episode_data, tracker_status, estatisticas_por_pista):
    """Calcula estatísticas gerais e totais"""
    episodios = episode_data['episodes']
    distancias = episode_data['distances']
    tempos = episode_data['times']
    passos = episode_data['steps']
    recompensas = episode_data['rewards']
    sucessos = episode_data['success']
    
    sucessos_int = [1 if s else 0 for s in sucessos]
    
    # Calcular totais gerais
    totais_gerais = {
        'total_episodios': len(episodios),
        'melhor_distancia': formatar_numero_float(max(distancias)) if distancias else 0,
        'maior_recompensa': formatar_numero_float(max(recompensas)) if recompensas else 0,
        'menor_distancia': formatar_numero_float(min(distancias)) if distancias else 0,
        'media_recompensa': formatar_numero_float(sum(recompensas) / len(recompensas)) if recompensas else 0,
        'media_distancia': formatar_numero_float(sum(distancias) / len(distancias)) if distancias else 0,
        'episodios_sucesso': sum(sucessos_int),
        'percentual_sucessos': formatar_numero_float(sum(sucessos_int) / len(sucessos_int)) if sucessos_int else 0,
        'total_passos': sum(passos),
        'total_tempo': formatar_numero_float(sum(tempos)),
    }
    
    # Calcular médias entre pistas
    pistas_validas = [p for p in estatisticas_por_pista.values() if p['total_episodios'] > 0]
    
    if pistas_validas:
        totais_gerais.update({
            'media_geral_recompensa': formatar_numero_float(sum(p['media_recompensa'] for p in pistas_validas) / len(pistas_validas)),
            'media_geral_distancia': formatar_numero_float(sum(p['media_distancia'] for p in pistas_validas) / len(pistas_validas)),
            'media_geral_sucessos': formatar_numero_float(sum(p['episodios_sucesso'] for p in pistas_validas) / len(pistas_validas)),
            'media_geral_percentual_sucessos': formatar_numero_float(sum(p['percentual_sucessos'] for p in pistas_validas) / len(pistas_validas)),
        })
    else:
        totais_gerais.update({
            'media_geral_recompensa': 0,
            'media_geral_distancia': 0,
            'media_geral_sucessos': 0,
            'media_geral_percentual_sucessos': 0,
        })
    
    return totais_gerais

def json_para_csv(json_file_path, csv_file_path):
    """Converte um arquivo JSON para CSV"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        episode_data = data['episode_data']
        tracker_status = data['tracker_status']
        session_info = data['session_info']
        
        # Calcular estatísticas por pista
        estatisticas_por_pista = calcular_estatisticas_por_pista(episode_data, tracker_status)
        estatisticas_gerais = calcular_estatisticas_gerais(episode_data, tracker_status, estatisticas_por_pista)
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            # Cabeçalho com estatísticas
            csvfile.write("# RESUMO ESTATÍSTICO - COMPARAÇÃO DE TREINAMENTOS\n")
            csvfile.write("##################################################\n")
            csvfile.write(f"# Arquivo: {os.path.basename(json_file_path)}\n")
            csvfile.write(f"# Robot: {session_info['robot']}\n")
            csvfile.write(f"# Algorithm: {session_info['algorithm']}\n")
            csvfile.write(f"# Environment: {session_info['environment']}\n")
            csvfile.write(f"# Data: {session_info.get('save_time', datetime.now().isoformat())}\n")
            csvfile.write("##################################################\n")
            
            # QUADRO DE RESUMO POR PISTA
            csvfile.write("# QUADRO DE RESUMO POR TIPO DE PISTA\n")
            
            # Definir ordem das pistas
            pistas_ordenadas = ['PR', 'PBA', 'PG', 'PRB', 'PRD', 'PRA', 'CC']
            pistas_presentes = [p for p in pistas_ordenadas if p in estatisticas_por_pista]
            
            # Cabeçalho do quadro
            metricas = [
                'Total Episódios',
                'Melhor Distancia',
                'Maior Recompensa', 
                'Menor Distância',
                'Média recompensa',
                'Média Distância',
                'Episodios de sucesso',
                'Percentual de Sucessos',
                'Total de Passos',
                'Total tempo (s)',
                'Primeiro episódio de Sucesso',
                'Soma passos até Primeiro Sucesso',
                'Tempo do Primeiro Sucesso (s)',
                'Número do Episódio mais rápido >9m',
                'Melhor Tempo do Percurso (9m) (s)',
                '% Sucessos depois de 9m'
            ]
            
            # Escrever cabeçalho das colunas
            header_row = ['Métrica'] + pistas_presentes + ['MÉDIA']
            csvfile.write(','.join(header_row) + '\n')
            
            # Escrever cada métrica
            for metrica in metricas:
                linha = [metrica]
                
                for pista in pistas_presentes:
                    stats = estatisticas_por_pista[pista]
                    
                    if metrica == 'Total Episódios':
                        valor = stats['total_episodios']
                    elif metrica == 'Melhor Distancia':
                        valor = stats['melhor_distancia']
                    elif metrica == 'Maior Recompensa':
                        valor = stats['maior_recompensa']
                    elif metrica == 'Menor Distância':
                        valor = stats['menor_distancia']
                    elif metrica == 'Média recompensa':
                        valor = stats['media_recompensa']
                    elif metrica == 'Média Distância':
                        valor = stats['media_distancia']
                    elif metrica == 'Episodios de sucesso':
                        valor = stats['episodios_sucesso']
                    elif metrica == 'Percentual de Sucessos':
                        valor = f"{stats['percentual_sucessos']*100:.2f}%"
                    elif metrica == 'Total de Passos':
                        valor = stats['total_passos']
                    elif metrica == 'Total tempo (s)':
                        valor = stats['total_tempo']
                    elif metrica == 'Primeiro episódio de Sucesso':
                        valor = stats['primeiro_episodio_sucesso'] or '0'
                    elif metrica == 'Soma passos até Primeiro Sucesso':
                        valor = stats['soma_passos_ate_primeiro_sucesso'] or '0'
                    elif metrica == 'Tempo do Primeiro Sucesso (s)':
                        valor = stats['tempo_primeiro_sucesso_episodio'] or '0'
                    elif metrica == 'Número do Episódio mais rápido >9m':
                        valor = stats['episodio_mais_rapido_9m'] or '0'
                    elif metrica == 'Melhor Tempo do Percurso (9m) (s)':
                        valor = stats['melhor_tempo_9m'] or '0'
                    elif metrica == '% Sucessos depois de 9m':
                        valor = f"{stats['percentual_sucessos_apos_primeiro']*100:.2f}%"
                    else:
                        valor = '0'
                    
                    linha.append(str(valor))
                
                # Adicionar média da coluna
                if metrica in ['Média recompensa', 'Média Distância', 'Percentual de Sucessos', '% Sucessos depois de 9m']:
                    if metrica == 'Média recompensa':
                        linha.append(str(estatisticas_gerais['media_geral_recompensa']))
                    elif metrica == 'Média Distância':
                        linha.append(str(estatisticas_gerais['media_geral_distancia']))
                    elif metrica == 'Percentual de Sucessos':
                        linha.append(f"{estatisticas_gerais['media_geral_percentual_sucessos']*100:.2f}%")
                    elif metrica == '% Sucessos depois de 9m':
                        # Calcular média dos percentuais de sucesso após primeiro sucesso
                        percentuais = [p['percentual_sucessos_apos_primeiro'] for p in estatisticas_por_pista.values() if p['percentual_sucessos_apos_primeiro'] > 0]
                        media_percentual = sum(percentuais) / len(percentuais) if percentuais else 0
                        linha.append(f"{media_percentual*100:.2f}%")
                else:
                    linha.append('')  
                
                csvfile.write(','.join(linha) + '\n')
            
            csvfile.write("##################################################\n")
            csvfile.write("# DADOS DETALHADOS DOS EPISÓDIOS\n")
            
            # Escrever cabeçalho dos dados detalhados
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