import json
import csv
import os
import glob

def formatar_numero(valor):
    """Formata números com 3 casas decimais"""
    try:
        return f"{float(valor):.3f}"
    except (ValueError, TypeError):
        return str(valor)

def json_para_csv(json_file_path, csv_file_path):
    """Converte um arquivo JSON para CSV de forma simplificada"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        episode_data = data['episode_data']
        
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # Cabeçalho
            headers = [
                'episodio', 'distancia', 'tempo', 'recompensa', 'passos', 'sucesso'
            ]
            writer.writerow(headers)
            
            # Dados dos episódios
            for i in range(len(episode_data['episodes'])):
                row = [
                    episode_data['episodes'][i],
                    formatar_numero(episode_data['distances'][i]),
                    formatar_numero(episode_data['times'][i]),
                    formatar_numero(episode_data['rewards'][i]),
                    episode_data['steps'][i],
                    1 if episode_data['success'][i] else 0
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
    
    for arquivo_json in arquivos_json:
        nome_base = os.path.splitext(arquivo_json)[0]
        arquivo_csv = nome_base + '.csv'
        
        if json_para_csv(arquivo_json, arquivo_csv):
            print(f"Convertido: {arquivo_json} -> {arquivo_csv}")

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