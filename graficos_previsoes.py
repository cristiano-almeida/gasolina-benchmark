"""
GRÁFICOS DE PREVISÕES - 3 MELHORES MODELOS POR ESCOPO
A partir dos CSVs gerados pelo bench17_final_completo.py

Gera:
- Nacional: Previsões SARIMAX, LSTM, THETA vs Real
- MG: Previsões ETS_no_trend, Exponential Smoothing, SARIMAX vs Real
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# Configurações
OUTPUT_DIR = "graficos_previsoes"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Caminho dos CSVs (ajuste conforme sua estrutura)
BASE_DIR = "benchmark_v17_resultados_finais/CSVs"

# Cores para os modelos
CORES = {
    'Real': 'black',
    'SARIMAX': '#2E86AB',
    'LSTM': '#A23B72',
    'THETA': '#F18F01',
    'ETS_no_trend': '#2E8B57',
    'Exponential_Smoothing': '#FF6B35',
}

# Estilos de linha
ESTILOS = {
    'Real': '-',
    'SARIMAX': '--',
    'LSTM': '-.',
    'THETA': ':',
    'ETS_no_trend': '--',
    'Exponential_Smoothing': '-.',
}

# ============================================================================
# 1. GRÁFICO NACIONAL - 3 MELHORES MODELOS
# ============================================================================

def grafico_nacional():
    """Gráfico com Real vs SARIMAX, LSTM, THETA"""
    
    # Carrega dados
    df = pd.read_csv(os.path.join(BASE_DIR, 'forecasts_Nacional.csv'))
    df = df.rename(columns={'Unnamed: 0': 'data'})
    df['data'] = pd.to_datetime(df['data'])
    
    # Seleciona colunas
    modelos = ['Real', 'SARIMAX', 'LSTM', 'THETA']
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for modelo in modelos:
        cor = CORES.get(modelo, 'gray')
        estilo = ESTILOS.get(modelo, '-')
        linewidth = 2 if modelo == 'Real' else 1.5
        marker = 'o' if modelo == 'Real' else None
        markersize = 4 if modelo == 'Real' else None
        
        ax.plot(df['data'], df[modelo], 
                label=modelo if modelo == 'Real' else f'{modelo} (previsto)',
                color=cor, linestyle=estilo, linewidth=linewidth,
                marker=marker, markersize=markersize, alpha=0.9)
    
    # Configurações do gráfico
    ax.set_xlabel('Data (Quinzenas - 2025)', fontsize=12)
    ax.set_ylabel('Preço (R$/litro)', fontsize=12)
    ax.set_title('Previsão do Preço da Gasolina - Nacional (Teste 2025)\nTop 3 Modelos: SARIMAX (0,72%), LSTM (0,75%), THETA (0,79%)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Formata o eixo x para mostrar as datas de forma legível
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', rotation=45)
    
    # Ajusta limites do eixo y (dá um espaço de 5% acima e abaixo)
    y_min = min(df[modelos].min()) - 0.1
    y_max = max(df[modelos].max()) + 0.1
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'nacional_top3_previsoes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfico salvo: nacional_top3_previsoes.png")

# ============================================================================
# 2. GRÁFICO MG - 3 MELHORES MODELOS
# ============================================================================

def grafico_mg():
    """Gráfico com Real vs ETS_no_trend, Exponential Smoothing, SARIMAX"""
    
    # Carrega dados
    df = pd.read_csv(os.path.join(BASE_DIR, 'forecasts_MG.csv'))
    df = df.rename(columns={'Unnamed: 0': 'data'})
    df['data'] = pd.to_datetime(df['data'])
    
    # Seleciona colunas
    modelos = ['Real', 'ETS_no_trend', 'Exponential_Smoothing', 'SARIMAX']
    
    # Cria figura
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for modelo in modelos:
        cor = CORES.get(modelo, 'gray')
        estilo = ESTILOS.get(modelo, '-')
        linewidth = 2 if modelo == 'Real' else 1.5
        marker = 'o' if modelo == 'Real' else None
        markersize = 4 if modelo == 'Real' else None
        
        # Ajusta nome para exibição
        if modelo == 'Exponential_Smoothing':
            nome_exibicao = 'Exponential Smoothing'
        elif modelo == 'ETS_no_trend':
            nome_exibicao = 'ETS (sem tendência)'
        else:
            nome_exibicao = modelo
        
        ax.plot(df['data'], df[modelo], 
                label=nome_exibicao if modelo == 'Real' else f'{nome_exibicao} (previsto)',
                color=cor, linestyle=estilo, linewidth=linewidth,
                marker=marker, markersize=markersize, alpha=0.9)
    
    # Configurações do gráfico
    ax.set_xlabel('Data (Quinzenas - 2025)', fontsize=12)
    ax.set_ylabel('Preço (R$/litro)', fontsize=12)
    ax.set_title('Previsão do Preço da Gasolina - Minas Gerais (Teste 2025)\nTop 3 Modelos: ETS (0,68%), Exponential Smoothing (0,78%), SARIMAX (0,92%)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Formata o eixo x
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.tick_params(axis='x', rotation=45)
    
    # Ajusta limites do eixo y
    y_min = min(df[modelos].min()) - 0.1
    y_max = max(df[modelos].max()) + 0.1
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'mg_top3_previsoes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfico salvo: mg_top3_previsoes.png")

# ============================================================================
# 3. GRÁFICO COMBINADO - COMPARAÇÃO DOS MELHORES MODELOS
# ============================================================================

def grafico_combinado():
    """
    Gráfico com 2 subplots: Nacional (melhor: SARIMAX) e MG (melhor: ETS_no_trend)
    """
    
    # Carrega dados
    df_nac = pd.read_csv(os.path.join(BASE_DIR, 'forecasts_Nacional.csv'))
    df_nac = df_nac.rename(columns={'Unnamed: 0': 'data'})
    df_nac['data'] = pd.to_datetime(df_nac['data'])
    
    df_mg = pd.read_csv(os.path.join(BASE_DIR, 'forecasts_MG.csv'))
    df_mg = df_mg.rename(columns={'Unnamed: 0': 'data'})
    df_mg['data'] = pd.to_datetime(df_mg['data'])
    
    # Cria figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ===== SUBPLOT 1: NACIONAL =====
    ax1.plot(df_nac['data'], df_nac['Real'], 'o-', label='Real', 
             color='black', linewidth=2, markersize=4)
    ax1.plot(df_nac['data'], df_nac['SARIMAX'], 's--', label='SARIMAX (previsto)\nMAPE=0,72%', 
             color='#2E86AB', linewidth=1.5, markersize=4)
    ax1.set_xlabel('Data (2025)', fontsize=10)
    ax1.set_ylabel('Preço (R$/litro)', fontsize=10)
    ax1.set_title('Nacional - Melhor Modelo: SARIMAX', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(axis='x', rotation=45)
    
    # ===== SUBPLOT 2: MINAS GERAIS =====
    ax2.plot(df_mg['data'], df_mg['Real'], 'o-', label='Real', 
             color='black', linewidth=2, markersize=4)
    ax2.plot(df_mg['data'], df_mg['ETS_no_trend'], 's--', label='ETS_no_trend (previsto)\nMAPE=0,68%', 
             color='#2E8B57', linewidth=1.5, markersize=4)
    ax2.set_xlabel('Data (2025)', fontsize=10)
    ax2.set_ylabel('Preço (R$/litro)', fontsize=10)
    ax2.set_title('Minas Gerais - Melhor Modelo: ETS_no_trend', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Comparação das Previsões dos Melhores Modelos vs Valores Reais (Teste 2025)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparativo_melhores_modelos.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfico salvo: comparativo_melhores_modelos.png")

# ============================================================================
# 4. GRÁFICO DE ERROS (RESÍDUOS) DOS MELHORES MODELOS
# ============================================================================

def grafico_erros_melhores():
    """Gráfico de erros (resíduos) dos melhores modelos"""
    
    # Carrega dados
    df_nac = pd.read_csv(os.path.join(BASE_DIR, 'forecasts_Nacional.csv'))
    df_nac = df_nac.rename(columns={'Unnamed: 0': 'data'})
    df_nac['data'] = pd.to_datetime(df_nac['data'])
    
    df_mg = pd.read_csv(os.path.join(BASE_DIR, 'forecasts_MG.csv'))
    df_mg = df_mg.rename(columns={'Unnamed: 0': 'data'})
    df_mg['data'] = pd.to_datetime(df_mg['data'])
    
    # Calcula resíduos
    residuos_nac = df_nac['Real'] - df_nac['SARIMAX']
    residuos_mg = df_mg['Real'] - df_mg['ETS_no_trend']
    
    # Cria figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # ===== NACIONAL =====
    ax1.bar(df_nac['data'], residuos_nac, width=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel('Data (2025)', fontsize=10)
    ax1.set_ylabel('Erro (R$/litro)', fontsize=10)
    ax1.set_title('Nacional - Erros do SARIMAX\nMAE = R$ 0,0450', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(axis='x', rotation=45)
    
    # ===== MINAS GERAIS =====
    ax2.bar(df_mg['data'], residuos_mg, width=10, color='darkorange', alpha=0.7, edgecolor='black')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel('Data (2025)', fontsize=10)
    ax2.set_ylabel('Erro (R$/litro)', fontsize=10)
    ax2.set_title('Minas Gerais - Erros do ETS_no_trend\nMAE = R$ 0,0420', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Erros das Previsões dos Melhores Modelos (Teste 2025)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'erros_melhores_modelos.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfico salvo: erros_melhores_modelos.png")

# ============================================================================
# 5. GRÁFICO COMPARATIVO - TODOS OS MODELOS EM UM ÚNICO GRÁFICO
# ============================================================================

def grafico_todos_modelos():
    """Gráfico com todos os modelos para cada escopo (para ver a dispersão)"""
    
    df_nac = pd.read_csv(os.path.join(BASE_DIR, 'forecasts_Nacional.csv'))
    df_nac = df_nac.rename(columns={'Unnamed: 0': 'data'})
    df_nac['data'] = pd.to_datetime(df_nac['data'])
    
    df_mg = pd.read_csv(os.path.join(BASE_DIR, 'forecasts_MG.csv'))
    df_mg = df_mg.rename(columns={'Unnamed: 0': 'data'})
    df_mg['data'] = pd.to_datetime(df_mg['data'])
    
    modelos = ['SARIMAX', 'XGBoost', 'LSTM', 'THETA', 'ETS_no_trend', 'Exponential_Smoothing', 'Naive']
    
    # Cores para cada modelo (mapa consistente)
    cores_modelos = {
        'SARIMAX': '#2E86AB', 'XGBoost': '#E63946', 'LSTM': '#A23B72',
        'THETA': '#F18F01', 'ETS_no_trend': '#2E8B57', 'Exponential_Smoothing': '#FF6B35', 'Naive': '#6C757D'
    }
    
    # Figura com 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # NACIONAL
    ax1.plot(df_nac['data'], df_nac['Real'], 'o-', label='Real', color='black', 
             linewidth=2.5, markersize=5, zorder=10)
    for modelo in modelos:
        if modelo in df_nac.columns:
            ax1.plot(df_nac['data'], df_nac[modelo], '--', label=modelo, 
                     color=cores_modelos.get(modelo, 'gray'), linewidth=1, alpha=0.7)
    ax1.set_xlabel('Data (2025)', fontsize=10)
    ax1.set_ylabel('Preço (R$/litro)', fontsize=10)
    ax1.set_title('Nacional - Previsões de Todos os Modelos', fontsize=11, fontweight='bold')
    ax1.legend(loc='best', fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.tick_params(axis='x', rotation=45)
    
    # MINAS GERAIS
    ax2.plot(df_mg['data'], df_mg['Real'], 'o-', label='Real', color='black', 
             linewidth=2.5, markersize=5, zorder=10)
    for modelo in modelos:
        if modelo in df_mg.columns:
            ax2.plot(df_mg['data'], df_mg[modelo], '--', label=modelo, 
                     color=cores_modelos.get(modelo, 'gray'), linewidth=1, alpha=0.7)
    ax2.set_xlabel('Data (2025)', fontsize=10)
    ax2.set_ylabel('Preço (R$/litro)', fontsize=10)
    ax2.set_title('Minas Gerais - Previsões de Todos os Modelos', fontsize=11, fontweight='bold')
    ax2.legend(loc='best', fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle('Comparação das Previsões: Todos os Modelos vs Valores Reais (Teste 2025)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'todos_modelos_previsoes.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✅ Gráfico salvo: todos_modelos_previsoes.png")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("GERANDO GRÁFICOS DE PREVISÕES A PARTIR DOS CSVs")
    print("="*60)
    
    # Verifica se os CSVs existem
    if not os.path.exists(BASE_DIR):
        print(f"\n❌ Pasta {BASE_DIR} não encontrada!")
        print("   Primeiro execute: python bench17_final_completo.py")
        return
    
    print("\n📊 Gerando gráficos...\n")
    
    grafico_nacional()
    grafico_mg()
    grafico_combinado()
    grafico_erros_melhores()
    grafico_todos_modelos()
    
    print("\n" + "="*60)
    print("✅ GRÁFICOS GERADOS COM SUCESSO!")
    print(f"📁 Pasta: {OUTPUT_DIR}/")
    print("\n📋 ARQUIVOS GERADOS:")
    print("   • nacional_top3_previsoes.png")
    print("   • mg_top3_previsoes.png")
    print("   • comparativo_melhores_modelos.png")
    print("   • erros_melhores_modelos.png")
    print("   • todos_modelos_previsoes.png")
    print("="*60)

if __name__ == "__main__":
    main()