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
    page_title="Quant Factor Lab Pro",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 0. UNIVERSO AMPLO & MAPA DE SETORES (MELHORIA: DADOS MAIS ROBUSTOS)
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

# Mapa manual para corrigir falhas do Yahoo Finance e garantir comparação justa
SECTOR_MAP_FIX = {
    'ITUB3.SA': 'Financial Services', 'ITUB4.SA': 'Financial Services',
    'BBDC3.SA': 'Financial Services', 'BBDC4.SA': 'Financial Services',
    'BBAS3.SA': 'Financial Services', 'SANB11.SA': 'Financial Services',
    'BPAC11.SA': 'Financial Services', 'B3SA3.SA': 'Financial Services',
    'BBSE3.SA': 'Financial Services', 'PSSA3.SA': 'Financial Services',
    'ITSA4.SA': 'Financial Services', 'CXSE3.SA': 'Financial Services',
    
    'VALE3.SA': 'Basic Materials', 'GGBR4.SA': 'Basic Materials', 
    'CSNA3.SA': 'Basic Materials', 'USIM5.SA': 'Basic Materials',
    'SUZB3.SA': 'Basic Materials', 'KLBN11.SA': 'Basic Materials',
    'BRAP4.SA': 'Basic Materials', 'UNIP6.SA': 'Basic Materials',
    
    'PETR3.SA': 'Energy', 'PETR4.SA': 'Energy', 'PRIO3.SA': 'Energy',
    'RRRP3.SA': 'Energy', 'CSAN3.SA': 'Energy', 'UGPA3.SA': 'Energy',
    'RAIZ4.SA': 'Energy',
    
    'ELET3.SA': 'Utilities', 'ELET6.SA': 'Utilities', 'CMIG4.SA': 'Utilities', 
    'CMIG3.SA': 'Utilities', 'CPLE6.SA': 'Utilities', 'CPFE3.SA': 'Utilities',
    'EGIE3.SA': 'Utilities', 'EQTL3.SA': 'Utilities', 'ENBR3.SA': 'Utilities',
    'TAEE11.SA': 'Utilities', 'TAEE3.SA': 'Utilities', 'TRPL4.SA': 'Utilities',
    'SBSP3.SA': 'Utilities', 'SAPR11.SA': 'Utilities', 'SAPR4.SA': 'Utilities',
    
    'VIVT3.SA': 'Communication Services', 'TIMS3.SA': 'Communication Services',
    
    'WEGE3.SA': 'Industrials', 'EMBR3.SA': 'Industrials', 'AZUL4.SA': 'Industrials',
    'GOLL4.SA': 'Industrials', 'CCRO3.SA': 'Industrials', 'ECOR3.SA': 'Industrials',
    'RAIL3.SA': 'Industrials', 'FRAS3.SA': 'Industrials',
    
    'MGLU3.SA': 'Consumer Cyclical', 'LREN3.SA': 'Consumer Cyclical', 
    'ARZZ3.SA': 'Consumer Cyclical', 'ALPA4.SA': 'Consumer Cyclical',
    'CVCB3.SA': 'Consumer Cyclical', 'PETZ3.SA': 'Consumer Cyclical',
    
    'ABEV3.SA': 'Consumer Defensive', 'BRFS3.SA': 'Consumer Defensive',
    'JBSS3.SA': 'Consumer Defensive', 'BEEF3.SA': 'Consumer Defensive',
    'MRFG3.SA': 'Consumer Defensive', 'MDIA3.SA': 'Consumer Defensive',
    'ASAI3.SA': 'Consumer Defensive', 'CRFB3.SA': 'Consumer Defensive',
    'SMTO3.SA': 'Consumer Defensive', 'AGRO3.SA': 'Consumer Defensive',
    
    'RADL3.SA': 'Healthcare', 'HAPV3.SA': 'Healthcare', 'FLRY3.SA': 'Healthcare',
    'RDOR3.SA': 'Healthcare',
    
    'TOTS3.SA': 'Technology', 'LWSA3.SA': 'Technology',
}

# ==============================================================================
# MÓDULO 1: DATA FETCHING
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca preços ajustados."""
    t_list = list(set(tickers)) # Remove duplicatas
    if 'BOVA11.SA' not in t_list:
        t_list.append('BOVA11.SA')
    
    try:
        data = yf.download(t_list, start=start_date, end=end_date, progress=False, auto_adjust=False)['Adj Close']
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Se baixou apenas 1 ticker, vira Series, converte para DF
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro no download de preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(user_tickers: list) -> pd.DataFrame:
    """Busca fundamentos do Universo Expandido + Tickers do Usuário."""
    
    # Combina listas
    combined_tickers = list(set(user_tickers) | set(BROAD_UNIVERSE))
    clean_tickers = [t for t in combined_tickers if t != 'BOVA11.SA']
    
    data = []
    # Barra de progresso para dar feedback visual
    progress_text = "Baixando Fundamentos do Mercado..."
    my_bar = st.progress(0, text=progress_text)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            # Tenta obter info
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            
            # 1. Tenta pegar setor do Yahoo
            y_sector = info.get('sector', info.get('Sector', 'Unknown'))
            
            # 2. Sobrescreve com o mapa manual se existir (Correção de Dados)
            final_sector = SECTOR_MAP_FIX.get(t, y_sector)
            
            data.append({
                'ticker': t,
                'Sector': final_sector,
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan),
                # Adicionado para filtro de liquidez futuro
                'marketCap': info.get('marketCap', np.nan) 
            })
        except:
            pass
        
        # Atualiza a cada 5 iterações para não travar a UI
        if i % 5 == 0:
            my_bar.progress((i + 1) / total, text=f"Baixando {t}...")
            
    my_bar.empty()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data).set_index('ticker')
    
    # Remove tickers que não têm dados essenciais (ex: P/L e P/VP vazios)
    # Isso limpa o universo de "sujeira"
    df.dropna(subset=['priceToBook', 'Sector'], inplace=True)
    
    return df

# ==============================================================================
# MÓDULO 2 & 3: FATORES E SCORING (MANTIDO E OTIMIZADO)
# ==============================================================================

def compute_factors_and_scores(prices, fundamentals, weights):
    # --- 1. Cálculos de Fatores ---
    
    # Momentum Residual
    res_mom = pd.Series(dtype=float)
    if not prices.empty and 'BOVA11.SA' in prices.columns:
        monthly = prices.resample('ME').last() # Pandas 2.2+ use 'ME'
        rets = monthly.pct_change().dropna()
        market = rets['BOVA11.SA']
        scores = {}
        for t in rets.columns:
            if t == 'BOVA11.SA': continue
            try:
                y = rets[t].tail(13)
                x = market.tail(13)
                if len(y) < 13: continue
                model = sm.OLS(y.values, sm.add_constant(x.values)).fit()
                resid = model.resid[:-1] # Remove ultimo mês (momentum clássico)
                scores[t] = np.sum(resid) / np.std(resid) if np.std(resid) > 0 else 0
            except:
                scores[t] = 0
        res_mom = pd.Series(scores, name='Res_Mom')

    # Momentum Fundamental
    fund_metrics = pd.DataFrame(index=fundamentals.index)
    if 'earningsGrowth' in fundamentals: fund_metrics['EG'] = fundamentals['earningsGrowth']
    if 'revenueGrowth' in fundamentals: fund_metrics['RG'] = fundamentals['revenueGrowth']
    fund_mom = fund_metrics.mean(axis=1).rename('Fund_Mom')

    # Value Score (Inverso P/L e P/VP)
    val_score = pd.DataFrame(index=fundamentals.index)
    if 'forwardPE' in fundamentals: 
        val_score['EP'] = np.where(fundamentals['forwardPE'] > 0, 1/fundamentals['forwardPE'], 0)
    if 'priceToBook' in fundamentals: 
        val_score['BP'] = np.where(fundamentals['priceToBook'] > 0, 1/fundamentals['priceToBook'], 0)
    val_final = val_score.mean(axis=1).rename('Value')

    # Quality Score
    qual_score = pd.DataFrame(index=fundamentals.index)
    if 'returnOnEquity' in fundamentals: qual_score['ROE'] = fundamentals['returnOnEquity']
    if 'profitMargins' in fundamentals: qual_score['PM'] = fundamentals['profitMargins']
    if 'debtToEquity' in fundamentals: 
        # Alavancagem segura
        qual_score['DE'] = -1 * np.where(fundamentals['debtToEquity'] > 0, fundamentals['debtToEquity'], 0)
    qual_final = qual_score.mean(axis=1).rename('Quality')

    # --- 2. Normalização Robusta ---
    
    df_scores = pd.DataFrame(index=fundamentals.index)
    df_scores['Sector'] = fundamentals['Sector']
    df_scores['Res_Mom'] = res_mom
    df_scores['Fund_Mom'] = fund_mom
    df_scores['Value'] = val_final
    df_scores['Quality'] = qual_final
    
    # Função Z-Score auxiliar
    def zscore(x):
        return (x - x.median()) / ((x - x.median()).abs().median() * 1.4826 + 1e-6)

    # Normalização
    cols_sector = ['Value', 'Quality']
    cols_global = ['Res_Mom', 'Fund_Mom']
    
    norm_cols = []
    
    # Setorial
    for c in cols_sector:
        if c in df_scores.columns:
            new_col = f"{c}_Z"
            # Agrupa por setor mapeado manualmente
            df_scores[new_col] = df_scores.groupby('Sector')[c].transform(zscore).clip(-3, 3)
            norm_cols.append(new_col)
            
    # Global
    for c in cols_global:
        if c in df_scores.columns:
            new_col = f"{c}_Z"
            df_scores[new_col] = zscore(df_scores[c]).clip(-3, 3)
            norm_cols.append(new_col)
            
    # Score Final Ponderado
    df_scores['Composite_Score'] = 0.0
    df_scores['Composite_Score'] += df_scores.get('Res_Mom_Z', 0).fillna(0) * weights['Res_Mom_Z']
    df_scores['Composite_Score'] += df_scores.get('Fund_Mom_Z', 0).fillna(0) * weights['Fund_Mom_Z']
    df_scores['Composite_Score'] += df_scores.get('Value_Z', 0).fillna(0) * weights['Value_Z']
    df_scores['Composite_Score'] += df_scores.get('Quality_Z', 0).fillna(0) * weights['Quality_Z']
    
    return df_scores.sort_values('Composite_Score', ascending=False), norm_cols

# ==============================================================================
# APP PRINCIPAL
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab Pro: Brasil Broad Market")
    st.markdown("Análise multifatorial com universo de comparação expandido (~90 ativos líquidos da B3).")

    # --- SIDEBAR ---
    st.sidebar.header("1. Seleção de Ativos")
    # Sugestão padrão
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
            # Datas
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback)
            
            # Fetch
            all_tickers_for_price = list(set(user_tickers) | set(BROAD_UNIVERSE))
            # Baixa preços de TODOS (broad) para calcular momentum relativo corretamente se necessário
            # (Aqui limitamos aos users para o gráfico, mas o momentum residual é melhor com user apenas vs ibov)
            prices = fetch_price_data(user_tickers, start_date, end_date)
            
            # Fundamentos AMPLOS (User + Broad) para normalização setorial
            fund_df = fetch_fundamentals(user_tickers)
            
            if fund_df.empty or prices.empty:
                st.error("Dados insuficientes.")
                return

            # Cálculo
            weights = {'Res_Mom_Z': w_rm, 'Fund_Mom_Z': w_fm, 'Value_Z': w_val, 'Quality_Z': w_qual}
            
            # Passamos prices apenas dos users para momentum, mas fundamentos de TODOS para normalização
            ranked_df_all, norm_cols = compute_factors_and_scores(prices, fund_df, weights)
            
            # Filtra apenas os tickers do usuário para exibição final
            # Mas preserva os Z-Scores que foram calculados usando o universo todo
            final_user_df = ranked_df_all.loc[ranked_df_all.index.intersection(user_tickers)].sort_values('Composite_Score', ascending=False)
            
            # Alocação (1 / Vol)
            sel_tickers = final_user_df.head(top_n).index.tolist()
            if sel_tickers:
                recent_vol = prices[sel_tickers].pct_change().std()
                inv_vol = 1 / (recent_vol + 1e-6)
                weights_alloc = inv_vol / inv_vol.sum()
            else:
                weights_alloc = pd.Series()

        # --- VISUALIZAÇÃO ---
        tab1, tab2, tab3 = st.tabs(["📊 Análise & Scatter", "🏆 Ranking & Alocação", "🔍 Detalhes"])
        
        with tab1:
            st.subheader("Mapa de Oportunidades: Value vs Quality")
            st.markdown("Este gráfico mostra onde seus ativos se posicionam em relação ao mercado (Universo Expandido). Buscamos o quadrante superior direito (Alta Qualidade e 'Barato').")
            
            # Plota TODO o universo para contexto
            fig_scatter = px.scatter(
                ranked_df_all.reset_index(), 
                x='Value_Z', 
                y='Quality_Z', 
                color='Sector',
                hover_data=['ticker', 'Composite_Score'],
                text='ticker',
                title="Value (Eixo X) vs Quality (Eixo Y) - Z-Scores Setoriais"
            )
            
            # Destaca os ativos do usuário
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

        with tab2:
            col1, col2 = st.columns([2,1])
            with col1:
                st.subheader("Top Picks (Sua Carteira)")
                st.dataframe(final_user_df[['Composite_Score', 'Sector'] + norm_cols].style.background_gradient(cmap='RdYlGn'))
            with col2:
                st.subheader("Alocação Sugerida (Risco Inverso)")
                if not weights_alloc.empty:
                    df_w = weights_alloc.to_frame("Peso")
                    df_w['Peso'] = df_w['Peso'].map("{:.2%}".format)
                    st.table(df_w)
                    
        with tab3:
            st.subheader("Comparação com Universo Amplo")
            st.dataframe(ranked_df_all[['Composite_Score', 'Sector'] + norm_cols])

if __name__ == "__main__":
    main()
