import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

def realizar_testes_t(dados_por_categoria, categorias_validas, metricas):
    """Realiza testes t para todas as métricas especificadas"""
    for metrica in metricas:
        tabela_teste_t = []
        
        for i, cat1 in enumerate(categorias_validas):
            for j, cat2 in enumerate(categorias_validas):
                if i < j:  # Evitar duplicatas e comparação com ela mesma
                    try:
                        dados1 = dados_por_categoria[cat1][metrica]
                        dados2 = dados_por_categoria[cat2][metrica]
                        
                        # Teste t independente
                        t_stat, p_value = stats.ttest_ind(dados1, dados2, equal_var=False)
                        
                        # Determinar significância
                        significativo = "Sim" if p_value < 0.05 else "Não"
                        
                        tabela_teste_t.append({
                            'Categoria_1': cat1,
                            'Categoria_2': cat2,
                            'Estatistica_t': f"{t_stat:.3f}",
                            'Valor_p': f"{p_value:.4f}",
                            'Significativo_α=0.05': significativo
                        })
                    except:
                        continue
        
        if tabela_teste_t:
            df_teste_t = pd.DataFrame(tabela_teste_t)
            df_teste_t.to_csv(f'teste_t_{metrica}.csv', index=False, encoding='utf-8-sig')

def analisar_ultimos_5000_episodios():
    """Analisa os últimos 5000 episódios de todos os arquivos CSV na pasta"""
    
    # Encontrar todos os arquivos CSV
    arquivos_csv = glob.glob("*.csv")
    
    if not arquivos_csv:
        return
    
    # Coletar dados dos últimos 5000 episódios de cada arquivo
    todos_dados = []
    
    for arquivo in arquivos_csv:
        try:
            df = pd.read_csv(arquivo)
            
            # Pegar últimos 5000 episódios (ou todos se tiver menos)
            ultimos_episodios = df.tail(5000)
            
            # Adicionar identificação do arquivo (categoria)
            nome_arquivo = os.path.splitext(arquivo)[0]
            ultimos_episodios['categoria'] = nome_arquivo
            
            todos_dados.append(ultimos_episodios)
            
        except Exception as e:
            continue
    
    if not todos_dados:
        return
    
    # Combinar todos os dados
    df_completo = pd.concat(todos_dados, ignore_index=True)
    
    # Lista de categorias para análise
    categorias = ['PR', 'PBA', 'PG', 'PRB', 'PRD', 'PRA', 'CC']
    
    # Dicionário para armazenar resultados
    resultados = {}
    dados_por_categoria = {}
    
    for categoria in categorias:
        df_cat = df_completo[df_completo['categoria'] == categoria]
        
        if len(df_cat) == 0:
            continue
            
        dados_por_categoria[categoria] = {
            'distancia': df_cat['distancia'],
            'tempo': df_cat['tempo'],
            'recompensa': df_cat['recompensa'],
            'passos': df_cat['passos'],
            'sucesso': df_cat['sucesso']
        }
        
        # Estatísticas básicas
        resultados[categoria] = {
            'distancia_media': df_cat['distancia'].mean(),
            'distancia_mediana': df_cat['distancia'].median(),
            'distancia_std': df_cat['distancia'].std(),
            
            'tempo_media': df_cat['tempo'].mean(),
            'tempo_mediana': df_cat['tempo'].median(),
            'tempo_std': df_cat['tempo'].std(),
            
            'recompensa_media': df_cat['recompensa'].mean(),
            'recompensa_mediana': df_cat['recompensa'].median(),
            'recompensa_std': df_cat['recompensa'].std(),
            
            'passos_media': df_cat['passos'].mean(),
            'passos_mediana': df_cat['passos'].median(),
            'passos_std': df_cat['passos'].std(),
            
            'sucesso_media': df_cat['sucesso'].mean() * 100,
            'sucesso_mediana': df_cat['sucesso'].median(),
            'sucesso_std': df_cat['sucesso'].std() * 100,
            
            'eficiencia_rp': (df_cat['recompensa'] / df_cat['passos']).mean(),
            'eficiencia_vt': (df_cat['distancia'] / df_cat['tempo']).mean(),
            'eficiencia_dp': (df_cat['distancia'] / df_cat['passos']).mean()
        }
    
    # TESTES T PARA TODAS AS MÉTRICAS
    categorias_validas = [cat for cat in categorias if cat in resultados]
    
    if len(categorias_validas) > 1:
        metricas_teste = ['distancia', 'tempo', 'recompensa', 'passos', 'sucesso']
        realizar_testes_t(dados_por_categoria, categorias_validas, metricas_teste)
    
    # TABELAS ESTATÍSTICAS BÁSICAS
    
    # Tabela de Distância
    tabela_distancia = []
    for cat in categorias:
        if cat in resultados:
            dados = resultados[cat]
            tabela_distancia.append([
                cat,
                f"{dados['distancia_media']:.1f}".replace('.', ','),
                f"{dados['distancia_mediana']:.0f}",
                f"{dados['distancia_std']:.1f}".replace('.', ',')
            ])
    
    df_distancia = pd.DataFrame(tabela_distancia, columns=['Distância', 'Média', 'Mediana', 'Desvio Padrão'])
    df_distancia.to_csv('tabela_distancia.csv', index=False, encoding='utf-8-sig')
    
    # Tabela de Tempo
    tabela_tempo = []
    for cat in categorias:
        if cat in resultados:
            dados = resultados[cat]
            tabela_tempo.append([
                cat,
                f"{dados['tempo_media']:.2f}".replace('.', ','),
                f"{dados['tempo_mediana']:.2f}".replace('.', ','),
                f"{dados['tempo_std']:.2f}".replace('.', ',')
            ])
    
    df_tempo = pd.DataFrame(tabela_tempo, columns=['Tempo', 'Média', 'Mediana', 'Desvio Padrão'])
    df_tempo.to_csv('tabela_tempo.csv', index=False, encoding='utf-8-sig')
    
    # Tabela de Recompensa
    tabela_recompensa = []
    for cat in categorias:
        if cat in resultados:
            dados = resultados[cat]
            media = f"{dados['recompensa_media']:,.1f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            mediana = f"{dados['recompensa_mediana']:,.1f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            std = f"{dados['recompensa_std']:,.1f}".replace(',', 'X').replace('.', ',').replace('X', '.')
            
            tabela_recompensa.append([cat, media, mediana, std])
    
    df_recompensa = pd.DataFrame(tabela_recompensa, columns=['Recompensa', 'Média', 'Mediana', 'Desvio Padrão'])
    df_recompensa.to_csv('tabela_recompensa.csv', index=False, encoding='utf-8-sig')
    
    # Tabela de Passos
    tabela_passos = []
    for cat in categorias:
        if cat in resultados:
            dados = resultados[cat]
            tabela_passos.append([
                cat,
                f"{dados['passos_media']:.1f}".replace('.', ','),
                f"{dados['passos_mediana']:.0f}",
                f"{dados['passos_std']:.1f}".replace('.', ',')
            ])
    
    df_passos = pd.DataFrame(tabela_passos, columns=['Passos', 'Média', 'Mediana', 'Desvio Padrão'])
    df_passos.to_csv('tabela_passos.csv', index=False, encoding='utf-8-sig')
    
    # Tabela de Sucessos
    tabela_sucessos = []
    for cat in categorias:
        if cat in resultados:
            dados = resultados[cat]
            tabela_sucessos.append([
                cat,
                f"{dados['sucesso_media']:.1f}%".replace('.', ','),
                f"{dados['sucesso_mediana']:.0f}",
                f"{dados['sucesso_std']:.1f}%".replace('.', ',')
            ])
    
    df_sucessos = pd.DataFrame(tabela_sucessos, columns=['Sucessos', 'Média', 'Mediana', 'Desvio Padrão'])
    df_sucessos.to_csv('tabela_sucessos.csv', index=False, encoding='utf-8-sig')
    
    # Tabela de Eficiência
    tabela_eficiencia = []
    for cat in categorias:
        if cat in resultados:
            dados = resultados[cat]
            tabela_eficiencia.append([
                cat,
                f"{dados['eficiencia_rp']:.2f}".replace('.', ',')
            ])
    
    df_eficiencia = pd.DataFrame(tabela_eficiencia, columns=['Eficiencia', 'Média'])
    df_eficiencia.to_csv('tabela_eficiencia.csv', index=False, encoding='utf-8-sig')
    
    # PAINEL CONSOLIDADO (sem consistência)
    painel_consolidado = []
    
    for cat in categorias:
        if cat in resultados:
            dados = resultados[cat]
            
            # Normalizar métricas para escala 0-100
            max_eficiencia_rp = max([resultados[c]['eficiencia_rp'] for c in resultados.keys()])
            max_sucesso = max([resultados[c]['sucesso_media'] for c in resultados.keys()])
            max_eficiencia_vt = max([resultados[c]['eficiencia_vt'] for c in resultados.keys()])
            max_eficiencia_dp = max([resultados[c]['eficiencia_dp'] for c in resultados.keys()])
            
            # Calcular scores individuais (0-100)
            score_eficiencia_rp = (dados['eficiencia_rp'] / max_eficiencia_rp) * 100 if max_eficiencia_rp > 0 else 0
            score_sucesso = (dados['sucesso_media'] / max_sucesso) * 100 if max_sucesso > 0 else 0
            score_eficiencia_vt = (dados['eficiencia_vt'] / max_eficiencia_vt) * 100 if max_eficiencia_vt > 0 else 0
            score_eficiencia_dp = (dados['eficiencia_dp'] / max_eficiencia_dp) * 100 if max_eficiencia_dp > 0 else 0
            
            # Calcular score final com pesos especificados
            peso_eficiencia_rp = 0.20
            peso_sucesso = 0.40
            peso_eficiencia_vt = 0.20
            peso_eficiencia_dp = 0.20
            
            score_final = (
                score_eficiencia_rp * peso_eficiencia_rp +
                score_sucesso * peso_sucesso +
                score_eficiencia_vt * peso_eficiencia_vt +
                score_eficiencia_dp * peso_eficiencia_dp
            )
            
            painel_consolidado.append({
                'Categoria': cat,
                'Eficiencia_R/P': f"{dados['eficiencia_rp']:.2f}".replace('.', ','),
                'Score_Eficiencia_R/P': f"{score_eficiencia_rp:.1f}",
                '%_Sucesso': f"{dados['sucesso_media']:.1f}%".replace('.', ','),
                'Score_%_Sucesso': f"{score_sucesso:.1f}",
                'Eficiencia_v/t': f"{dados['eficiencia_vt']:.3f}".replace('.', ','),
                'Score_Eficiencia_v/t': f"{score_eficiencia_vt:.1f}",
                'Eficiencia_d/p': f"{dados['eficiencia_dp']:.4f}".replace('.', ','),
                'Score_Eficiencia_d/p': f"{score_eficiencia_dp:.1f}",
                'Score_Final': f"{score_final:.1f}"
            })
    
    df_painel = pd.DataFrame(painel_consolidado)
    
    # Ordenar por score final
    df_painel['Score_Final_Num'] = df_painel['Score_Final'].str.replace(',', '.').astype(float)
    df_painel = df_painel.sort_values('Score_Final_Num', ascending=False)
    df_painel = df_painel.drop('Score_Final_Num', axis=1)
    
    df_painel.to_csv('painel_consolidado.csv', index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    analisar_ultimos_5000_episodios()