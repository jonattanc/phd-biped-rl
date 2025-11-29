import pandas as pd
import numpy as np
from scipy import stats
import os
import glob

def separar_cc_por_pista(df_cc):
    """Separa o arquivo CC.csv por pista baseado na ordem alternada"""
    # Ordem das pistas: PBA, PG, PRA, PRB, PRD, PR
    pistas = ['PBA', 'PG', 'PRA', 'PRB', 'PRD', 'PR']
    
    # Criar DataFrames separados para cada pista
    dados_por_pista = {}
    
    for i, pista in enumerate(pistas):
        # Selecionar episódios desta pista (cada 6 episódios, começando pela posição i)
        episodios_pista = df_cc.iloc[i::6].copy()
        episodios_pista['categoria'] = f'CC_{pista}'
        dados_por_pista[pista] = episodios_pista
    
    return dados_por_pista

def realizar_testes_t(dados_por_categoria, categorias_validas, metricas, sufixo=""):
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
            nome_arquivo = f'teste_t_{metrica}{sufixo}.csv'
            df_teste_t = pd.DataFrame(tabela_teste_t)
            df_teste_t.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')

def analisar_ultimos_5000_episodios():
    """Analisa os últimos 5000 episódios de todos os arquivos CSV na pasta"""
    
    # Encontrar todos os arquivos CSV
    arquivos_csv = glob.glob("*.csv")
    
    if not arquivos_csv:
        return
    
    # Coletar dados dos últimos 5000 episódios de cada arquivo
    todos_dados = []
    todos_dados_cc_separado = []
    
    for arquivo in arquivos_csv:
        try:
            df = pd.read_csv(arquivo)
            nome_arquivo = os.path.splitext(arquivo)[0]
            
            # Processar todos os arquivos normalmente (incluindo CC como método geral)
            ultimos_episodios = df.tail(5000)
            ultimos_episodios['categoria'] = nome_arquivo
            todos_dados.append(ultimos_episodios)
            
            # Se for o arquivo CC, também separar por pista para análises detalhadas
            if nome_arquivo == 'CC':
                dados_cc_separados = separar_cc_por_pista(df)
                
                # Adicionar cada pista do CC separadamente
                for pista, df_pista in dados_cc_separados.items():
                    ultimos_episodios_pista = df_pista.tail(5000)
                    ultimos_episodios_pista['categoria'] = f'CC_{pista}'
                    todos_dados_cc_separado.append(ultimos_episodios_pista)
            
        except Exception as e:
            continue
    
    # ANÁLISE 1: Métodos Gerais (CC como método único)
    if todos_dados:
        df_completo_geral = pd.concat(todos_dados, ignore_index=True)
        analisar_dados_gerais(df_completo_geral)
    
    # ANÁLISE 2: CC Separado por Pista (comparação detalhada)
    if todos_dados_cc_separado:
        df_completo_cc_separado = pd.concat(todos_dados_cc_separado, ignore_index=True)
        analisar_cc_por_pista(df_completo_cc_separado)

def analisar_dados_gerais(df_completo):
    """Analisa dados considerando CC como método geral"""
    
    # Lista de categorias para análise (métodos gerais)
    categorias_gerais = ['PR', 'PBA', 'PG', 'PRB', 'PRD', 'PRA', 'CC']
    
    # Dicionário para armazenar resultados
    resultados_gerais = {}
    dados_por_categoria_geral = {}
    
    for categoria in categorias_gerais:
        df_cat = df_completo[df_completo['categoria'] == categoria]
        
        if len(df_cat) == 0:
            continue
            
        dados_por_categoria_geral[categoria] = {
            'distancia': df_cat['distancia'],
            'tempo': df_cat['tempo'],
            'recompensa': df_cat['recompensa'],
            'passos': df_cat['passos'],
            'sucesso': df_cat['sucesso']
        }
        
        # Estatísticas básicas
        resultados_gerais[categoria] = calcular_estatisticas_basicas(df_cat)
    
    # TESTES T PARA MÉTODOS GERAIS
    categorias_validas_geral = [cat for cat in categorias_gerais if cat in resultados_gerais]
    
    if len(categorias_validas_geral) > 1:
        metricas_teste = ['distancia', 'tempo', 'recompensa', 'passos', 'sucesso']
        realizar_testes_t(dados_por_categoria_geral, categorias_validas_geral, metricas_teste, "_geral")
    
    # GERAR TABELAS PARA MÉTODOS GERAIS
    gerar_tabelas_estatisticas(resultados_gerais, categorias_gerais, "_geral")
    
    # PAINEL CONSOLIDADO PARA MÉTODOS GERAIS
    gerar_painel_consolidado(resultados_gerais, categorias_gerais, "painel_consolidado_geral.csv")

def analisar_cc_por_pista(df_completo_cc):
    """Analisa CC separado por pista para comparação detalhada"""
    
    # Lista de categorias para análise CC por pista
    categorias_cc = ['CC_PBA', 'CC_PG', 'CC_PRA', 'CC_PRB', 'CC_PRD', 'CC_PR']
    
    # Dicionário para armazenar resultados
    resultados_cc = {}
    dados_por_categoria_cc = {}
    
    for categoria in categorias_cc:
        df_cat = df_completo_cc[df_completo_cc['categoria'] == categoria]
        
        if len(df_cat) == 0:
            continue
            
        dados_por_categoria_cc[categoria] = {
            'distancia': df_cat['distancia'],
            'tempo': df_cat['tempo'],
            'recompensa': df_cat['recompensa'],
            'passos': df_cat['passos'],
            'sucesso': df_cat['sucesso']
        }
        
        # Estatísticas básicas
        resultados_cc[categoria] = calcular_estatisticas_basicas(df_cat)
    
    # TESTES T PARA CC POR PISTA
    categorias_validas_cc = [cat for cat in categorias_cc if cat in resultados_cc]
    
    if len(categorias_validas_cc) > 1:
        metricas_teste = ['distancia', 'tempo', 'recompensa', 'passos', 'sucesso']
        realizar_testes_t(dados_por_categoria_cc, categorias_validas_cc, metricas_teste, "_cc_pistas")
    
    # GERAR TABELAS PARA CC POR PISTA
    gerar_tabelas_estatisticas(resultados_cc, categorias_cc, "_cc_pistas")
    
    # PAINEL CONSOLIDADO PARA CC POR PISTA
    gerar_painel_consolidado(resultados_cc, categorias_cc, "painel_consolidado_cc_pistas.csv")

def calcular_estatisticas_basicas(df_cat):
    """Calcula estatísticas básicas para um DataFrame"""
    return {
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

def gerar_tabelas_estatisticas(resultados, categorias, sufixo):
    """Gera tabelas estatísticas básicas"""
    
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
    df_distancia.to_csv(f'tabela_distancia{sufixo}.csv', index=False, encoding='utf-8-sig')
    
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
    df_tempo.to_csv(f'tabela_tempo{sufixo}.csv', index=False, encoding='utf-8-sig')
    
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
    df_recompensa.to_csv(f'tabela_recompensa{sufixo}.csv', index=False, encoding='utf-8-sig')
    
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
    df_passos.to_csv(f'tabela_passos{sufixo}.csv', index=False, encoding='utf-8-sig')
    
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
    df_sucessos.to_csv(f'tabela_sucessos{sufixo}.csv', index=False, encoding='utf-8-sig')

def gerar_painel_consolidado(resultados, categorias, nome_arquivo):
    """Gera painel consolidado"""
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
    
    df_painel.to_csv(nome_arquivo, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
    analisar_ultimos_5000_episodios()