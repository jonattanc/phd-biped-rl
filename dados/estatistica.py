import pandas as pd
import numpy as np
import glob
import os

def calcular_metricas_gerais():
    """Calcula métricas gerais para cada arquivo CSV"""
    
    arquivos_csv = glob.glob("*.csv")
    
    # Remover arquivos de resultados anteriores para não analisá-los
    arquivos_csv = [f for f in arquivos_csv if not f.startswith(('tabela_', 'estatisticas_', 'analise_'))]
    
    if not arquivos_csv:
        print("Nenhum arquivo CSV encontrado para análise.")
        return
    
    resultados = []
    
    for arquivo in arquivos_csv:
        try:
            df = pd.read_csv(arquivo)
            nome_categoria = os.path.splitext(arquivo)[0]
            
            # Garantir que as colunas numéricas estão no formato correto
            df['distancia'] = pd.to_numeric(df['distancia'], errors='coerce')
            df['tempo'] = pd.to_numeric(df['tempo'], errors='coerce')
            df['recompensa'] = pd.to_numeric(df['recompensa'], errors='coerce')
            df['passos'] = pd.to_numeric(df['passos'], errors='coerce')
            df['sucesso'] = pd.to_numeric(df['sucesso'], errors='coerce')
            
            # Remover valores NaN
            df = df.dropna(subset=['distancia', 'tempo', 'recompensa', 'passos', 'sucesso'])
            
            if len(df) == 0:
                continue
            
            # 1. Total Episódios
            total_episodios = len(df)
            
            # 2. Nº do Primeiro Episódio ≥ 9m
            episodios_9m = df[df['distancia'] >= 9.0]
            primeiro_episodio_9m = episodios_9m['episodio'].iloc[0] if len(episodios_9m) > 0 else None
            
            # 3. Total de Passos
            total_passos = df['passos'].sum()
            
            # 4. Soma Passos Até Atingir ≥ 9m
            if primeiro_episodio_9m is not None:
                passos_ate_9m = df[df['episodio'] <= primeiro_episodio_9m]['passos'].sum()
            else:
                passos_ate_9m = total_passos
            
            # 5. Total de Passos Até Parar de Evoluir (2500000)
            passos_ate_estabilizar = min(total_passos, 2500000)
            
            # 6. Passos de Aprendizagem Após 1° 9m
            if primeiro_episodio_9m is not None:
                passos_apos_9m = total_passos - passos_ate_9m
            else:
                passos_apos_9m = 0
            
            # 7. Média Distância
            media_distancia = df['distancia'].mean()
            
            # 8. Desvio padrão da média percorrida
            desvio_padrao_distancia = df['distancia'].std()
            
            # 9. Maior Recompensa
            maior_recompensa = df['recompensa'].max()
            
            # 10. Tempo Total do Treinamento (s)
            tempo_total = df['tempo'].sum()
            
            # 11. Melhor Tempo dos Episódios ≥ 9m
            if len(episodios_9m) > 0:
                melhor_tempo_9m = episodios_9m['tempo'].min()
            else:
                melhor_tempo_9m = None
            
            # 12. Número de Sucessos ≥ 9m
            sucessos_9m = len(episodios_9m[episodios_9m['sucesso'] == 1])
            
            # 13. % Sucessos Pós-aprendizagem ≥ 9m
            if primeiro_episodio_9m is not None:
                episodios_pos_9m = df[df['episodio'] > primeiro_episodio_9m]
                if len(episodios_pos_9m) > 0:
                    percentual_sucessos_pos = (episodios_pos_9m['sucesso'].sum() / len(episodios_pos_9m)) * 100
                else:
                    percentual_sucessos_pos = 0
            else:
                percentual_sucessos_pos = 0
            
            # 14. Velocidade média de sucessos
            episodios_sucesso = df[df['sucesso'] == 1]
            if len(episodios_sucesso) > 0:
                velocidade_media_sucessos = (episodios_sucesso['distancia'].sum() / episodios_sucesso['tempo'].sum())
            else:
                velocidade_media_sucessos = 0
            
            # 15. Eficiencia RP (Recompensa/Passo após 9m)
            if primeiro_episodio_9m is not None and passos_apos_9m > 0:
                recompensa_apos_9m = episodios_pos_9m['recompensa'].sum()
                eficiencia_rp = recompensa_apos_9m / passos_apos_9m
            else:
                eficiencia_rp = 0
            
            # 16. Eficiencia DP (Distância/Passo após 9m)
            if primeiro_episodio_9m is not None and passos_apos_9m > 0:
                distancia_apos_9m = episodios_pos_9m['distancia'].sum()
                eficiencia_dp = distancia_apos_9m / passos_apos_9m
            else:
                eficiencia_dp = 0
            
            # 17. Eficiencia VT (Distância/Tempo após 9m)
            if primeiro_episodio_9m is not None:
                tempo_apos_9m = episodios_pos_9m['tempo'].sum()
                if tempo_apos_9m > 0:
                    eficiencia_vt = distancia_apos_9m / tempo_apos_9m
                else:
                    eficiencia_vt = 0
            else:
                eficiencia_vt = 0
            
            # Coletar resultados
            resultado = {
                'Categoria': nome_categoria,
                'Total_Episodios': int(total_episodios),
                'Primeiro_Episodio_9m': primeiro_episodio_9m if primeiro_episodio_9m is not None else 'N/A',
                'Total_Passos': int(total_passos),
                'Passos_Ate_9m': int(passos_ate_9m),
                'Passos_Ate_Estabilizar': int(passos_ate_estabilizar),
                'Passos_Apos_9m': int(passos_apos_9m),
                'Media_Distancia': round(media_distancia, 3),
                'Desvio_Padrao_Distancia': round(desvio_padrao_distancia, 3),
                'Maior_Recompensa': round(maior_recompensa, 3),
                'Tempo_Total_Treinamento': round(tempo_total, 3),
                'Melhor_Tempo_9m': round(melhor_tempo_9m, 3) if melhor_tempo_9m is not None else 'N/A',
                'Sucessos_9m': int(sucessos_9m),
                'Percentual_Sucessos_Pos_9m': round(percentual_sucessos_pos, 3),
                'Velocidade_Media_Sucessos': round(velocidade_media_sucessos, 3),
                'Eficiencia_RP': round(eficiencia_rp, 3),
                'Eficiencia_DP': round(eficiencia_dp, 3),
                'Eficiencia_VT': round(eficiencia_vt, 3)
            }
            
            resultados.append(resultado)
            
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")
            continue
    
    if resultados:
        # Criar DataFrame com resultados
        df_resultados = pd.DataFrame(resultados)
        
        # Exportar para CSV
        df_resultados.to_csv('estatisticas_gerais.csv', index=False, encoding='utf-8-sig')
        
        # Gerar tabela formatada para visualização
        gerar_tabelas_formatadas(df_resultados)
        
        print("Análise concluída! Arquivo 'estatisticas_gerais.csv' gerado com sucesso.")
    else:
        print("Nenhum resultado válido encontrado.")

def gerar_tabelas_formatadas(df_resultados):
    """Gera tabelas formatadas a partir dos resultados"""
    
    # Tabela resumo geral
    tabela_resumo = []
    
    for _, row in df_resultados.iterrows():
        tabela_resumo.append([
            row['Categoria'],
            row['Total_Episodios'],
            row['Primeiro_Episodio_9m'],
            f"{row['Total_Passos']:,}".replace(',', '.'),
            f"{row['Passos_Ate_9m']:,}".replace(',', '.'),
            f"{row['Passos_Ate_Estabilizar']:,}".replace(',', '.'),
            f"{row['Passos_Apos_9m']:,}".replace(',', '.'),
            f"{row['Media_Distancia']:.3f}".replace('.', ','),
            f"{row['Desvio_Padrao_Distancia']:.3f}".replace('.', ','),
            f"{row['Maior_Recompensa']:,.3f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
            f"{row['Tempo_Total_Treinamento']:,.1f}".replace(',', 'X').replace('.', ',').replace('X', '.'),
            row['Melhor_Tempo_9m'] if row['Melhor_Tempo_9m'] != 'N/A' else 'N/A',
            row['Sucessos_9m'],
            f"{row['Percentual_Sucessos_Pos_9m']:.1f}%".replace('.', ','),
            f"{row['Velocidade_Media_Sucessos']:.3f}".replace('.', ','),
            f"{row['Eficiencia_RP']:.3f}".replace('.', ','),
            f"{row['Eficiencia_DP']:.3f}".replace('.', ','),
            f"{row['Eficiencia_VT']:.3f}".replace('.', ',')
        ])
    
    colunas_resumo = [
        'Categoria', 'Total Episódios', '1° Episódio ≥9m', 'Total Passos', 
        'Passos até 9m', 'Passos até Estabilizar', 'Passos após 9m',
        'Média Distância', 'Desvio Padrão Distância', 'Maior Recompensa', 
        'Tempo Total (s)', 'Melhor Tempo ≥9m', 'Sucessos ≥9m', 
        '% Sucessos Pós-9m', 'Velocidade Média', 'Eficiência RP', 
        'Eficiência DP', 'Eficiência VT'
    ]
    
    df_tabela_resumo = pd.DataFrame(tabela_resumo, columns=colunas_resumo)
    df_tabela_resumo.to_csv('tabela_resumo_geral.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    calcular_metricas_gerais()