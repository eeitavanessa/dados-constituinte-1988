# -*- coding: utf-8 -*-
"""
ANÁLISE EXPLORATÓRIA - SUGESTÕES PARA A CONSTITUINTE 1986
Autor: Vanessa Pedroso
Data: outubro/2025
"""

import pandas as pd 
import numpy as np # para operações que envolvem números
import matplotlib.pyplot as plt # para criar visualizações mais elaboradas
import seaborn as sns # para criar visualizações pré-definidas
from collections import Counter
import re # busca e substituição de expressões regulares
import os # manipulação do diretório local

# Configurações para melhor visualização no VSCode - parâmetros de visualização
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
sns.set_style("whitegrid")

print("🚀 INICIANDO ANÁLISE DOS DADOS DA CONSTITUINTE...\n")

def carregar_dados():
    """Carrega e prepara o dataset"""
    try:
        # Verifica se o arquivo existe
        if not os.path.exists('dados_constituinte.csv'):
            print("❌ Arquivo 'dados_constituinte.csv' não encontrado!")
            print("📁 Certifique-se de que o arquivo está na mesma pasta do script")
            return None
        
        df = pd.read_csv('dados_constituinte.csv', delimiter=';', encoding='latin-1', na_values=['NA', ''])
        print(f"✅ Dataset carregado com sucesso!")
        print(f"📊 Total de registros: {len(df):,}")
        print(f"📈 Total de colunas: {len(df.columns)}")
        return df
    
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return None

def analise_preliminar(df):
    """Análise inicial dos dados"""
    print("\n" + "="*50)
    print("📋 ANÁLISE PRELIMINAR")
    print("_"*50)
    
    # Primeiras linhas
    print("\n🔍 Primeiras 5 linhas:")
    print(df.head())
    
    # Informações das colunas
    print("\n📝 Colunas disponíveis:")
    for i, coluna in enumerate(df.columns, 1):
        print(f"  {i:2d}. {coluna}")
    
    # Valores missing
    print("\n📉 Valores faltantes:")
    missing = df.isnull().sum()
    for coluna, faltantes in missing[missing > 0].items():
        percentual = (faltantes / len(df)) * 100
        print(f"  • {coluna}: {faltantes} ({percentual:.1f}%)")

def analise_demografica(df):
    """Análise do perfil demográfico dos participantes"""
    print("\n" + "="*50)
    print("👥 ANÁLISE DEMOGRÁFICA")
    print("="*50)
    
    # Criar figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('PERFIL DEMOGRÁFICO DOS PARTICIPANTES', fontsize=16, fontweight='bold')
    
    # 1. Distribuição por Sexo
    df['SEXO'] = df['SEXO'].fillna('NÃO INFORMADO')
    sexo_counts = df['SEXO'].value_counts()
    colors_sexo = ["#331212", '#4ECDC4', '#95A5A6']  # Vermelho, Verde, Cinza
    axes[0,0].pie(sexo_counts.values, labels=sexo_counts.index, autopct='%1.1f%%', 
                  colors=colors_sexo, startangle=90)
    axes[0,0].set_title('Distribuição por Sexo', fontweight='bold')
    
    # 2. Distribuição por Faixa Etária
    df['FAIXA ETÁRIA'] = df['FAIXA ETÁRIA'].fillna('NÃO INFORMADO')
    faixa_etaria = df['FAIXA ETÁRIA'].value_counts()
    # Reordenar para melhor visualização
    ordem_faixa = ['15 A 19 ANOS', '20 A 24 ANOS', '25 A 29 ANOS', '30 A 39 ANOS', 
                   '40 A 49 ANOS', '50 A 59 ANOS', 'ACIMA DE 59 ANOS', 'NÃO INFORMADO']
    faixa_etaria = faixa_etaria.reindex(ordem_faixa, fill_value=0)
    
    bars = axes[0,1].bar(faixa_etaria.index, faixa_etaria.values, color='skyblue', alpha=0.8)
    axes[0,1].set_title('Distribuição por Faixa Etária', fontweight='bold')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            axes[0,1].text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}', ha='center', va='bottom')
    
    # 3. Distribuição por Escolaridade
    df['INSTRUCAO'] = df['INSTRUCAO'].fillna('NÃO INFORMADO')
    instrucao = df['INSTRUCAO'].value_counts().head(8)
    bars = axes[1,0].barh(instrucao.index, instrucao.values, color='lightgreen', alpha=0.8)
    axes[1,0].set_title('Distribuição por Escolaridade', fontweight='bold')
    
    # Adicionar valores nas barras horizontais
    for bar in bars:
        width = bar.get_width()
        axes[1,0].text(width, bar.get_y() + bar.get_height()/2.,
                     f' {int(width)}', ha='left', va='center')
    
    # 4. Distribuição por Estado Civil
    df['ESTADO CIVIL'] = df['ESTADO CIVIL'].fillna('NÃO INFORMADO')
    estado_civil = df['ESTADO CIVIL'].value_counts().head(6)
    colors_estado = ['#FF9FF3', '#F368E0', '#FF9F43', '#10AC84', '#54A0FF', '#5F27CD']
    axes[1,1].pie(estado_civil.values, labels=estado_civil.index, autopct='%1.1f%%',
                  colors=colors_estado, startangle=90)
    axes[1,1].set_title('Distribuição por Estado Civil', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('perfil_demografico.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Estatísticas detalhadas
    print(f"\n📊 ESTATÍSTICAS DETALHADAS:")
    print(f"• Homens: {len(df[df['SEXO'] == 'MASCULINO']):,} ({(len(df[df['SEXO'] == 'MASCULINO'])/len(df))*100:.1f}%)")
    print(f"• Mulheres: {len(df[df['SEXO'] == 'FEMININO']):,} ({(len(df[df['SEXO'] == 'FEMININO'])/len(df))*100:.1f}%)")
    print(f"• Sexo não informado: {len(df[df['SEXO'] == 'NÃO INFORMADO']):,}")

def analise_geografica(df):
    """Análise da distribuição geográfica"""
    print("\n" + "="*50)
    print("🗺️ ANÁLISE GEOGRÁFICA")
    print("="*50)
    
    df['UF'] = df['UF'].fillna('NÃO INFORMADO')
    uf_distribuicao = df['UF'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    colors = plt.cm.Set3(np.linspace(0, 1, len(uf_distribuicao)))
    bars = plt.bar(uf_distribuicao.index, uf_distribuicao.values, color=colors)
    
    plt.title('TOP 10 ESTADOS COM MAIS SUGESTÕES', fontweight='bold', fontsize=14)
    plt.xlabel('Estado', fontweight='bold')
    plt.ylabel('Número de Sugestões', fontweight='bold')
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig('distribuicao_geografica.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🏆 TOP 5 ESTADOS MAIS ENGAGADOS:")
    for i, (estado, count) in enumerate(uf_distribuicao.head().items(), 1):
        percentual = (count / len(df)) * 100
        print(f"  {i}. {estado}: {count:,} sugestões ({percentual:.1f}%)")

def analise_temporal(df):
    """Análise da evolução temporal"""
    print("\n" + "="*50)
    print("📅 ANÁLISE TEMPORAL")
    print("="*50)
    
    # Converter datas
    df['DATA'] = pd.to_datetime(df['DATA'], dayfirst=True, errors='coerce')
    
    # Agrupar por mês
    sugestoes_por_mes = df.groupby(df['DATA'].dt.to_period('M')).size()
    
    plt.figure(figsize=(14, 6))
    plt.plot(sugestoes_por_mes.index.astype(str), sugestoes_por_mes.values, 
             marker='o', linewidth=2, markersize=6, color='#6A0572', alpha=0.8)
    
    plt.title('EVOLUÇÃO TEMPORAL DAS SUGESTÕES', fontweight='bold', fontsize=14)
    plt.xlabel('Mês/Ano', fontweight='bold')
    plt.ylabel('Número de Sugestões', fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Destacar o pico
    mes_pico = sugestoes_por_mes.idxmax()
    valor_pico = sugestoes_por_mes.max()
    pico_index = list(sugestoes_por_mes.index).index(mes_pico)
    
    plt.annotate(f'Pico: {valor_pico} sugestões', 
                xy=(pico_index, valor_pico), 
                xytext=(pico_index, valor_pico + 10),
                arrowprops=dict(arrowstyle='->', color='red'),
                ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('evolucao_temporal.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📈 Mês com mais sugestões: {mes_pico} ({valor_pico} sugestões)")

def analise_conteudo(df):
    """Análise do conteúdo das sugestões"""
    print("\n" + "="*50)
    print("📝 ANÁLISE DE CONTEÚDO")
    print("="*50)
    
    # Juntar todos os textos
    textos = df['SUGESTAO.TEXTO'].dropna().astype(str)
    todos_textos = ' '.join(textos)
    
    # Análise de palavras
    palavras = re.findall(r'\b[a-záéíóúâêîôûãõç]{4,}\b', todos_textos.lower())
    
    # Stop words em português
    stop_words = {
        'que', 'com', 'para', 'uma', 'mais', 'como', 'sobre', 'seus', 'este', 'esta',
        'ser', 'seja', 'são', 'mas', 'muito', 'nosso', 'nossa', 'pelos', 'pelas',
        'essa', 'esse', 'isso', 'aquele', 'aquela', 'entre', 'através', 'quando'
    }
    
    palavras_filtradas = [p for p in palavras if p not in stop_words]
    contagem = Counter(palavras_filtradas)
    top_palavras = contagem.most_common(15)
    
    plt.figure(figsize=(12, 8))
    palavras, frequencias = zip(*top_palavras)
    
    bars = plt.barh(palavras, frequencias, color='#2E86AB', alpha=0.8)
    plt.title('15 PALAVRAS MAIS FREQUENTES NAS SUGESTÕES', fontweight='bold', fontsize=14)
    plt.xlabel('Frequência', fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Adicionar valores
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2., 
                f' {int(width)}', ha='left', va='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('palavras_frequentes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🔤 TOP 10 PALAVRAS-CHAVE:")
    for i, (palavra, freq) in enumerate(top_palavras[:10], 1):
        print(f"  {i}. {palavra.upper()}: {freq} ocorrências")

def resumo_final(df):
    """Gera um resumo final da análise"""
    print("\n" + "="*60)
    print("📊 RESUMO FINAL DA ANÁLISE")
    print("="*60)
    
    # Estatísticas principais
    total_sugestoes = len(df)
    participantes_masculinos = len(df[df['SEXO'] == 'MASCULINO'])
    participantes_femininos = len(df[df['SEXO'] == 'FEMININO'])
    
    # Estado mais ativo
    estado_mais_ativo = df['UF'].value_counts().index[0]
    sugestoes_estado_mais_ativo = df['UF'].value_counts().iloc[0]
    
    # Faixa etária mais comum
    faixa_mais_comum = df['FAIXA ETÁRIA'].value_counts().index[0]
    
    print(f"\n🎯 PRINCIPAIS ESTATÍSTICAS:")
    print(f"  • Total de sugestões analisadas: {total_sugestoes:,}")
    print(f"  • Participação masculina: {participantes_masculinos:,} ({(participantes_masculinos/total_sugestoes)*100:.1f}%)")
    print(f"  • Participação feminina: {participantes_femininos:,} ({(participantes_femininos/total_sugestoes)*100:.1f}%)")
    print(f"  • Estado mais engajado: {estado_mais_ativo} ({sugestoes_estado_mais_ativo:,} sugestões)")
    print(f"  • Faixa etária predominante: {faixa_mais_comum}")
    
    print(f"\n📈 GRÁFICOS GERADOS:")
    print("  ✅ perfil_demografico.png")
    print("  ✅ distribuicao_geografica.png") 
    print("  ✅ evolucao_temporal.png")
    print("  ✅ palavras_frequentes.png")
    
    print(f"\n💡 INSIGHTS INICIAIS:")
    print("  • [Seu insight 1 aqui]")
    print("  • [Seu insight 2 aqui]")
    print("  • [Seu insight 3 aqui]")

# EXECUÇÃO PRINCIPAL
if __name__ == "__main__":
    print("🔍 ANALISANDO DADOS DA CONSTITUINTE DE 1986")
    print("="*50)
    
    # Carregar dados
    df = carregar_dados()
    
    if df is not None:
        # Executar análises
        analise_preliminar(df)
        analise_demografica(df)
        analise_geografica(df)
        analise_temporal(df)
        analise_conteudo(df)
        resumo_final(df)
        
        print("\n🎉 ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("📁 Os gráficos foram salvos como arquivos PNG")
        
    else:
        print("❌ Não foi possível carregar os dados. Verifique o arquivo CSV.")
