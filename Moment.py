import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab Pro: Brasil Broad Market",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 0. UNIVERSO AMPLO & MAPA DE SETORES (Melhorado e Estável)
# ==============================================================================

# Universo expandido (~90 ativos mais líquidos da B3 para gerar estatísticas robustas)
BROAD_UNIVERSE = [
    'RRRP3.SA', 'ALOS3.SA', 'ALPA4.SA', 'ABEV3.SA', 'ARZZ3.SA', 'ASAI3.SA', 'AZUL4.SA', 'B3SA3.SA', 
    'BBSE3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BRAP4.SA', 'BBAS3.SA', 'BRKM5.SA', 'BRFS3.SA', 'BPAC11.SA', 
    'CRFB3.SA', 'CCRO3.SA', 'CMIG4.SA', 'CIEL3.SA', 'COGN3.SA', 'CPLE6.SA', 'CSAN3.SA', 'CPFE3.SA', 
    'CMIN3.SA', 'CVCB3.SA', 'CYRE3.SA', 'DXCO3.SA', 'ECOR3.SA', 'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 
    'ENBR3.SA', 'ENGI11.SA', 'ENEV3.SA', 'EGIE3.SA', 'EQTL3.SA', 'EZTC3.SA', 'FLRY3.SA', 'GGBR4.SA', 
    'GOAU4.SA', 'GOLL4.SA', 'HAPV3.SA', 'HYPE3.SA', 'IGTI11.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 
    'ITUB3.SA', 'JBSS3.SA', 'KLBN11.SA', 'RENT3.SA', 'LREN3.SA', 'LWSA3.SA', 'MGLU3.SA', 'MRFG3.SA', 
    'BEEF3.SA', 'MRVE3.SA', 'MULT3.SA', 'NTCO3.SA', 'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'PRIO3.SA', 
    'PETZ3.SA', 'RADL3.SA', 'RAIZ4.SA', 'RDOR3.SA', 'RAIL3.SA', 'SBSP3.SA', 'SANB11.SA', 'SMTO3.SA', 
    'CSNA3.SA', 'SLCE3.SA', 'SUZB3.SA', 'TAEE11.SA', 'TAEE3.SA', 'VIVT3.SA', 'TIMS3.SA', 'TOTS3.SA', 
    'UGPA3.SA', 'USIM5.SA', 'VALE3.SA', 'WEGE3.SA', 'YDUQ3.SA', 'SAPR11.SA', 'SAPR4.SA', 'PSSA3.SA',
    'TRPL4.SA', 'MDIA3.SA', 'FRAS3.SA', 'UNIP6.SA', 'AGRO3.SA', 'CXSE3.SA'
]

# Mapa manual para corrigir setores e garantir comparação justa
SECTOR_MAP_FIX = {
    'ITUB3.SA': 'Financial Services', 'ITUB4.SA': 'Financial Services',
    'BBDC3.SA': 'Financial Services', 'BBDC4.SA': 'Financial Services',
    'BBAS3.SA': 'Financial Services', 'SANB11.SA': 'Financial Services',
    'BPAC11.SA': 'Financial Services', 'B3SA3.SA': 'Financial Services',
    'BBSE3.SA': 'Financial Services', 'PSSA3.SA': 'Financial Services',
    'ITSA4.SA': 'Financial Services', 'CXSE3.SA': 'Financial Services',
    'VALE3.SA': 'Basic Materials', 'SBSP3.SA': 'Utilities',
    'SAPR4.SA': 'Utilities', 'SAPR11.SA': 'Utilities',
    'TAEE11.SA': 'Utilities', 'TAEE3.SA': 'Utilities',
    'MDIA3.SA': 'Consumer Defensive', 'FRAS3.SA': 'Industrials',
    'VIVT3.SA': 'Communication Services', 'PETR4.SA': 'Energy',
    # Adicionar mais ativos de saneamento/utilities
    'CMIG3.SA': 'Utilities', 'CMIG4.SA': 'Utilities', 'CPLE6.SA': 'Utilities',
}

# ==============================================================================
# MÓDULO 1: DATA FETCHING
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca preços ajustados."""
    t_list = list(set(tickers)) 
    if 'BOVA11.SA' not in t_list:
        t_list.append('BOVA11.SA')
    
    try:
        data = yf.download(t_list, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro no download de preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(user_tickers: list) -> pd.DataFrame:
    """Busca fundamentos do Universo Expandido + Tickers do Usuário."""
    
    combined_tickers = list(set(user_tickers) | set(BROAD_UNIVERSE))
    clean_tickers = [t for t in combined_tickers if t != 'BOVA11.SA']
    
    data = []
    my_bar = st.progress(0, text="Baixando Fundamentos do Mercado...")
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            info = yf.Ticker(t).info
            
            y_sector = info.get('sector', info.get('Sector', 'Unknown'))
            final_sector = SECTOR_MAP_FIX.get(t, y_sector)
            
            # ATENÇÃO: enterpriseToEbitda foi removido para corrigir o KeyError
            data.append({
                'ticker': t,
                'Sector': final_sector,
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan),
                'marketCap': info.get('marketCap', np.nan) 
            })
        except:
            pass
        
        if i % 5 == 0:
            my_bar.progress((i + 1) / total, text=f"Baixando {t}...")
            
    my_bar.empty()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data).set_index('ticker')
    # Remove linhas sem P/VP ou Setor (dados de baixa qualidade)
    df.dropna(subset=['priceToBook', 'Sector'], inplace=True)
    
    return df

# ==============================================================================
# MÓDULO 2 & 3: FATORES E SCORING
# ==============================================================================

def compute_factors_and_scores(prices, fundamentals, weights):
    # --- 1. Cálculos de Fatores ---
    
    # Momentum Residual
    res_mom = pd.Series(dtype=float)
    if not prices.empty and 'BOVA11.SA' in prices.columns:
        monthly = prices.resample('ME').last()
        rets = monthly.pct_change().dropna()
        market = rets['BOVA11.SA']
        scores = {}
        # Calculo de Beta e Residuo
        for t in rets.columns:
            if t == 'BOVA11.SA': continue
            try:
                # Regressão linear (janela de 13 meses, pula o último mês)
                y = rets[t].tail(13)
                x = market.tail(13)
                if len(y) < 13: continue
                model = sm.OLS(y.values, sm.add_constant(x.values)).fit()
                resid = model.resid[:-1] 
                scores[t] = np.sum(resid) / np.std(resid) if np.std(resid) > 0 else 0
            except:
                scores[t] = 0
        res_mom = pd.Series(scores, name='Res_Mom')

    # Momentum Fundamental (Média do Crescimento de Lucro e Receita)
    fund_metrics = pd.DataFrame(index=fundamentals.index)
    if 'earningsGrowth' in fundamentals: fund_metrics['EG'] = fundamentals['earningsGrowth']
    if 'revenueGrowth' in fundamentals: fund_metrics['RG'] = fundamentals['revenueGrowth']
    fund_mom = fund_metrics.mean(axis=1).rename('Fund_Mom')

    # Value Score (1/P/L e 1/P/VP)
    val_score = pd.DataFrame(index=fundamentals.index)
    if 'forwardPE' in fundamentals: 
        val_score['EP'] = np.where(fundamentals['forwardPE'] > 0, 1/fundamentals['forwardPE'], 0)
    if 'priceToBook' in fundamentals: 
        val_score['BP'] = np.where(fundamentals['priceToBook'] > 0, 1/fundamentals['priceToBook'], 0)
    val_final = val_score.mean(axis=1).rename('Value')

    # Quality Score (ROE, Margem, Inverso da Dívida)
    qual_score = pd.DataFrame(index=fundamentals.index)
    if 'returnOnEquity' in fundamentals: qual_score['ROE'] = fundamentals['returnOnEquity']
    if 'profitMargins' in fundamentals: qual_score['PM'] = fundamentals['profitMargins']
    if 'debtToEquity' in fundamentals: 
        qual_score['DE'] = -1 * np.where(fundamentals['debtToEquity'] > 0, fundamentals['debtToEquity'], 0)
    qual_final = qual_score.mean(axis=1).rename('Quality')

    # --- 2. Normalização Robusta ---
    
    df_scores = pd.DataFrame(index=fundamentals.index)
    df_scores['Sector'] = fundamentals['Sector']
    df_scores['Res_Mom'] = res_mom
    df_scores['Fund_Mom'] = fund_mom
    df_scores['Value'] = val_final
    df_scores['Quality'] = qual_final
    
    # Função Z-Score auxiliar (Robusto: Mediana e MAD)
    def zscore(x):
        # Evita divisão por zero com um pequeno epsilon
        mad = (x - x.median()).abs().median()
        return (x - x.median()) / (mad * 1.4826 + 1e-6)

    cols_sector = ['Value', 'Quality']
    cols_global = ['Res_Mom', 'Fund_Mom']
    norm_cols = []
    
    # Normalização Setorial (vs. pares do mesmo setor)
    for c in cols_sector:
        if c in df_scores.columns:
            new_col = f"{c}_Z"
            df_scores[new_col] = df_scores.groupby('Sector')[c].transform(zscore).clip(-3, 3)
            norm_cols.append(new_col)
            
    # Normalização Global (vs. mercado total)
    for c in cols_global:
        if c in df_scores.columns:
            new_col = f"{c}_Z"
            df_scores[new_col] = zscore(df_scores[c]).clip(-3, 3)
            norm_cols.append(new_col)
            
    # Score Final Ponderado
    df_scores['Composite_Score'] = 0.0
    for factor, weight in weights.items():
         df_scores['Composite_Score'] += df_scores.get(factor, 0).fillna(0) * weight
    
    return df_scores.sort_values('Composite_Score', ascending=False), norm_cols

# ==============================================================================
# MÓDULO 4: RISCO VS RETORNO (Melhoria)
# ==============================================================================

def compute_risk_return(prices_df: pd.DataFrame, tickers: list, annualization_factor=252):
    """Calcula Retorno e Volatilidade Anualizada."""
    
    # Inclui BOVA11.SA no benchmark
    t_list = [t for t in tickers if t in prices_df.columns]
    if 'BOVA11.SA' in prices_df.columns:
        t_list.append('BOVA11.SA')
    
    if not t_list: return pd.DataFrame()

    df = prices_df[t_list].copy()
    returns = df.pct_change().dropna()
    
    # Retorno Anualizado (média geométrica)
    cumulative_returns = (1 + returns).prod()**(annualization_factor / len(returns)) - 1
    
    # Volatilidade Anualizada
    annualized_volatility = returns.std() * np.sqrt(annualization_factor)
    
    stats = pd.DataFrame({
        'Retorno Anualizado': cumulative_returns,
        'Volatilidade Anualizada': annualized_volatility
    }).loc[t_list]
    
    return stats


# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    # ... (Configuração e Sidebar)
    st.title("🧪 Quant Factor Lab Pro: Brasil Broad Market")
    st.markdown("Análise multifatorial com universo de comparação expandido (~90 ativos líquidos da B3).")

    # --- SIDEBAR ---
    st.sidebar.header("1. Seleção de Ativos")
    default_univ = "ITUB3.SA, BBAS3.SA, VALE3.SA, WEGE3.SA, PRIO3.SA, TAEE11.SA, MDIA3.SA, FRAS3.SA, SAPR4.SA, VIVT3.SA"
    ticker_input = st.sidebar.text_area("Seus Tickers (Carteira)", default_univ, height=80)
    user_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.30)
    w_fm = st.sidebar.slider("Fundamental Momentum", 0.0, 1.0, 0.20)
    w_val = st.sidebar.slider("Value (Setorial)", 0.0, 1.0, 0.25)
    w_qual = st.sidebar.slider("Quality (Setorial)", 0.0, 1.0, 0.25)

    st.sidebar.header("3. Parâmetros")
    top_n = st.sidebar.number_input("Top N Ativos", 1, 20, 5)
    lookback = st.sidebar.selectbox("Histórico (Dias)", [252, 504, 756], index=1)
    
    run_btn = st.sidebar.button("🚀 Executar Análise", type="primary")


    if run_btn:
        if not user_tickers:
            st.warning("Insira tickers.")
            return
            
        with st.spinner("Processando dados de mercado..."):
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback)
            
            # 1. Fetch
            prices = fetch_price_data(user_tickers, start_date, end_date)
            fund_df = fetch_fundamentals(user_tickers)
            
            if fund_df.empty or prices.empty:
                st.error("Dados insuficientes.")
                return

            # 2. Cálculo de Fatores e Scores
            weights = {'Res_Mom_Z': w_rm, 'Fund_Mom_Z': w_fm, 'Value_Z': w_val, 'Quality_Z': w_qual}
            ranked_df_all, norm_cols = compute_factors_and_scores(prices, fund_df, weights)
            
            # Filtra apenas os tickers do usuário
            final_user_df = ranked_df_all.loc[ranked_df_all.index.intersection(user_tickers)].sort_values('Composite_Score', ascending=False)
            
            # 3. Alocação (1 / Vol)
            sel_tickers = final_user_df.head(top_n).index.tolist()
            if sel_tickers:
                recent_vol = prices[sel_tickers].pct_change().std()
                inv_vol = 1 / (recent_vol + 1e-6)
                weights_alloc = inv_vol / inv_vol.sum()
            else:
                weights_alloc = pd.Series()
                
            # 4. Risco/Retorno
            risk_return_df = compute_risk_return(prices, user_tickers)
            
            # Junta com os setores
            risk_return_df = risk_return_df.join(fund_df[['Sector']], how='left')


        # --- VISUALIZAÇÃO ---
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Análise Risco/Retorno", "📈 Value vs Quality", "🏆 Ranking & Alocação", "🔍 Detalhes"])
        
        with tab1:
            st.subheader(f"Histórico: Risco vs. Retorno (Últimos {int(lookback/21)} Meses)")
            st.markdown("Busque por ativos no quadrante superior esquerdo (Alto Retorno, Baixo Risco). A linha diagonal traçada pelo BOVA11.SA indica o *trade-off* do mercado.")
            
            if not risk_return_df.empty:
                # Plota BOVA11 em destaque
                bova_df = risk_return_df.loc[['BOVA11.SA']] if 'BOVA11.SA' in risk_return_df.index else pd.DataFrame()
                
                # Plota todos os ativos
                fig_risk = px.scatter(
                    risk_return_df.reset_index(),
                    x='Volatilidade Anualizada',
                    y='Retorno Anualizado',
                    color='Sector',
                    hover_data={'ticker': True, 'Retorno Anualizado': ':.2%', 'Volatilidade Anualizada': ':.2%'},
                    text='ticker',
                    title="Risco vs Retorno Anualizado"
                )
                
                fig_risk.update_traces(textposition='top center')
                
                # Adiciona o ponto do BOVA11.SA em destaque
                if not bova_df.empty:
                    fig_risk.add_scatter(
                        x=bova_df['Volatilidade Anualizada'],
                        y=bova_df['Retorno Anualizado'],
                        mode='markers+text',
                        marker=dict(color='red', size=15, symbol='star'),
                        text=['BOVA11.SA'],
                        textposition='bottom center',
                        name='Benchmark (BOVA11.SA)'
                    )
                
                st.plotly_chart(fig_risk, use_container_width=True)
            else:
                st.warning("Não foi possível calcular o Risco vs. Retorno.")


        with tab2:
            st.subheader("Mapa de Oportunidades: Value vs Quality (Z-Scores Setoriais)")
            st.markdown("Este gráfico compara o Valor e a Qualidade de **todos os ativos** do universo expandido. Seus ativos selecionados estão em destaque (marcadores maiores). Busque o quadrante superior direito (Qualidade e Desconto).")
            
            # Plota TODO o universo para contexto
            fig_scatter = px.scatter(
                ranked_df_all.reset_index(), 
                x='Value_Z', 
                y='Quality_Z', 
                color='Sector',
                hover_data={'ticker': True, 'Composite_Score': ':.2f'},
                text='ticker',
                title="Value (Desconto) vs Quality (Rentabilidade) - Z-Scores Setoriais"
            )
            
            # Destaca os ativos do usuário (pontos maiores)
            user_subset = ranked_df_all.loc[ranked_df_all.index.intersection(user_tickers)].reset_index()
            fig_scatter.add_scatter(
                x=user_subset['Value_Z'], 
                y=user_subset['Quality_Z'], 
                mode='markers+text', 
                marker=dict(color='black', size=12, symbol='star'),
                text=user_subset['ticker'],
                textposition='top center',
                name='Sua Carteira'
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)


        with tab3:
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader(f"Top {top_n} Picks por Score Composto")
                st.dataframe(final_user_df[['Composite_Score', 'Sector'] + norm_cols].head(top_n).style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']).format("{:.2f}"))
            with col2:
                st.subheader("Alocação Sugerida (Risco Inverso)")
                if not weights_alloc.empty:
                    df_w = weights_alloc.to_frame("Peso")
                    df_w['Peso'] = df_w['Peso'].map("{:.2%}".format)
                    st.table(df_w)
                    
        with tab4:
            st.subheader("Justificativa Detalhada da Seleção")
            st.markdown("A pontuação positiva indica desempenho **acima da média** de seus pares setoriais ou do mercado amplo.")

            detailed_df = final_user_df.head(top_n)
            
            for ticker in detailed_df.index:
                row = detailed_df.loc[ticker]
                st.markdown(f"### 📈 {ticker} - Score: {row.get('Composite_Score', 0):.2f}")
                st.markdown(f"**Setor:** {row.get('Sector', 'N/A')}")
                
                justification = []
                
                def analyze_factor(factor_name, label_name, high_desc, low_desc):
                    val = row.get(factor_name, 0)
                    if pd.isna(val): return
                    if val > 0.5: justification.append(f"- **{label_name} ({val:.2f}):** {high_desc}")
                    elif val > 0.0: justification.append(f"- **{label_name} ({val:.2f}):** Acima da média.")
                    else: justification.append(f"- **{label_name} ({val:.2f}):** {low_desc}")

                analyze_factor('Res_Mom_Z', 'Momentum Preço', 'Forte Alpha (retorno acima do Ibov).', 'Preço sem força relativa recente.')
                analyze_factor('Fund_Mom_Z', 'Momentum Fundamental', 'Lucros/Receita acelerando forte (topo do mercado).', 'Crescimento estagnado/negativo vs mercado.')
                analyze_factor('Value_Z', 'Valor (vs Setor)', 'Muito descontado (P/L e P/VP baixos) vs pares.', 'Preço justo ou caro vs pares setoriais.')
                analyze_factor('Quality_Z', 'Qualidade (vs Setor)', 'Alta eficiência (ROE) e solidez vs pares.', 'Qualidade na média ou abaixo dos pares.')

                st.info("\n".join(justification) if justification else "Sem dados suficientes para justificativa.")

if __name__ == "__main__":
    main()
