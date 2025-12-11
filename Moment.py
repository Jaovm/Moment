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
    page_title="Quant Factor Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# MÓDULO 1: DATA FETCHING (Busca de Dados)
# ==============================================================================

@st.cache_data(ttl=3600*12)
def fetch_price_data(tickers: list, start_date: str, end_date: str) -> pd.DataFrame:
    """Busca histórico de preços ajustados."""
    # Garante IBOV11.SA para cálculo de beta/mercado
    t_list = list(tickers)
    if 'IBOV11.SA' not in t_list:
        t_list.append('IBOV11.SA')
    
    try:
        # CORREÇÃO DO FUTURE WARNING: Definir auto_adjust=False para manter 
        # a coluna 'Adj Close' e evitar o aviso.
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False  # Mantém o comportamento original
        )['Adj Close']
        
        # Correção para o MultiIndex se for o caso
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """Busca snapshots fundamentais atuais."""
    data = []
    clean_tickers = [t for t in tickers if t != 'IBOV11.SA']
    
    # Barra de progresso para melhor UX
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            stock = yf.Ticker(t)
            info = stock.info
            
            # Coleta defensiva (se não existir, retorna NaN)
            data.append({
                'ticker': t,
                'sector': info.get('sector', 'Unknown'),
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan)
            })
        except:
            pass
        progress_bar.progress((i + 1) / total)
        
    progress_bar.empty()
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data).set_index('ticker')

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """
    Calcula Momentum Residual (Retorno idiossincrático vs IBOV11.SA).
    Metodologia: Regressão OLS de 12 meses. Score = Sum(Resíduos) / Std(Resíduos).
    """
    df = price_df.copy()
    # Resample mensal
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'IBOV11.SA' not in rets.columns:
        return pd.Series(dtype=float)
        
    market = rets['IBOV11.SA']
    scores = {}
    window = lookback + skip
    
    for ticker in rets.columns:
        if ticker == 'IBOV11.SA': continue
        
        y = rets[ticker].tail(window)
        x = market.tail(window)
        
        if len(y) < window:
            continue
            
        # Regressão: Ri = alpha + beta*Rm + erro
        X = sm.add_constant(x.values)
        model = sm.OLS(y.values, X).fit()
        
        # Pega resíduos excluindo o mês mais recente (skip)
        resid = model.resid[:-skip]
        
        # Score = Information Ratio dos resíduos
        if np.std(resid) == 0 or len(resid) < 2:
            scores[ticker] = 0
        else:
            scores[ticker] = np.sum(resid) / np.std(resid)
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_fundamental_momentum(fund_df: pd.DataFrame) -> pd.Series:
    """Z-Score combinado de crescimento de Receita e Lucro."""
    metrics = ['earningsGrowth', 'revenueGrowth']
    temp_df = pd.DataFrame(index=fund_df.index)
    
    for m in metrics:
        if m in fund_df.columns:
            s = fund_df[m].fillna(fund_df[m].median())
            temp_df[m] = (s - s.mean()) / s.std()
            
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor: Inverso de P/E e P/B."""
    scores = pd.DataFrame(index=fund_df.index)
    
    if 'forwardPE' in fund_df:
        scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
        
    if 'priceToBook' in fund_df:
        scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
        
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    
    if 'returnOnEquity' in fund_df: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df: scores['DE_Inv'] = -1 * fund_df['debtToEquity']
        
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """
    Z-Score Robusto usando Mediana e MAD (Median Absolute Deviation).
    Clipper em +/- 3 para remover outliers extremos.
    """
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    
    if mad == 0:
        return series - median
    
    z = (series - median) / (mad * 1.4826)
    return z.clip(-3, 3)

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Calcula score final ponderado."""
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    
    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
            
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 4: PORTFOLIO & BACKTEST
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    """Define pesos do portfólio (Equal Weight ou Vol Target)."""
    selected = ranked_df.head(top_n).index.tolist()
    
    if not selected:
        return pd.Series()

    if vol_target is not None:
        # Volatilidade recente (3 meses)
        recent_rets = prices[selected].pct_change().tail(63)
        vols = recent_rets.std() * (252**0.5)
        
        # Ajuste para ativos com volatilidade zero ou nula
        vols[vols == 0] = 1 # Evita divisão por zero
        
        # Peso inverso à volatilidade: w = (Target / Vol)
        raw_weights = vol_target / vols
        weights = raw_weights / raw_weights.sum()
    else:
        # Pesos iguais
        weights = pd.Series(1.0/len(selected), index=selected)
        
    return weights

def run_backtest(weights: pd.Series, prices: pd.DataFrame, lookback_days: int = 252):
    """
    Simula o desempenho do portfólio selecionado no período recente.
    """
    valid_tickers = [t for t in weights.index if t in prices.columns]
    if not valid_tickers:
        return pd.DataFrame()
        
    # Recorta o período
    subset = prices[valid_tickers].tail(lookback_days)
    rets = subset.pct_change().dropna()
    
    # Retorno do portfólio
    port_ret = rets.dot(weights[valid_tickers])
    
    # Cumulative Return (Equity Curve)
    cumulative = (1 + port_ret).cumprod()
    cumulative.name = "Strategy"
    
    # Benchmark (IBOV11.SA) se disponível
    if 'IBOV11.SA' in prices.columns:
        IBOV11.SA_ret = prices['IBOV11.SA'].tail(lookback_days).pct_change().dropna()
        IBOV11.SA_cum = (1 + IBOV11.SA_ret).cumprod()
        
        # Alinha datas
        combined = pd.DataFrame({'Strategy': cumulative, 'IBOV11.SA': IBOV11.SA_cum}).ffill().dropna()
        return combined
    
    return cumulative.to_frame()

# ==============================================================================
# APP PRINCIPAL (STREAMLIT UI)
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Multi-Strategy Engine")
    st.markdown("---")

    # --- SIDEBAR ---
    st.sidebar.header("1. Universo e Dados")
    default_univ = "AAPL, MSFT, GOOGL, AMZN, NVDA, TSLA, META, JPM, V, JNJ, PG, XOM, UNH, HD, MA, PFE, KO, PEP, MRK, AVGO, CSCO, MCD, ABT"
    ticker_input = st.sidebar.text_area("Tickers (Separados por vírgula)", default_univ, height=100)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_fm = st.sidebar.slider("Fundamental Momentum", 0.0, 1.0, 0.20)
    w_val = st.sidebar.slider("Value", 0.0, 1.0, 0.20)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Construção de Portfólio")
    top_n = st.sidebar.number_input("Número de Ativos (Top N)", 1, 20, 5)
    use_vol_target = st.sidebar.checkbox("Usar Volatility Targeting?", False)
    target_vol = st.sidebar.slider("Volatilidade Alvo (Anual)", 0.05, 0.30, 0.15) if use_vol_target else None
    
    run_btn = st.sidebar.button("🚀 Rodar Análise", type="primary")

    # --- MAIN LOGIC ---
    if run_btn:
        if not tickers:
            st.error("Por favor, insira pelo menos um ticker.")
            return

        with st.status("Executando Pipeline Quant...", expanded=True) as status:
            
            # 1. Dados
            st.write("📥 Baixando dados de mercado e fundamentais...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            prices = fetch_price_data(tickers, start_date, end_date)
            fundamentals = fetch_fundamentals(tickers)
            
            if prices.empty or fundamentals.empty:
                st.error("Não foi possível obter dados suficientes.")
                status.update(label="Erro!", state="error")
                return

            # 2. Cálculos
            st.write("🧮 Calculando fatores multifatoriais...")
            res_mom = compute_residual_momentum(prices)
            fund_mom = compute_fundamental_momentum(fundamentals)
            val_score = compute_value_score(fundamentals)
            qual_score = compute_quality_score(fundamentals)

            # 3. Consolidação
            st.write("⚖️ Normalizando e Ranking...")
            df_master = pd.DataFrame(index=tickers)
            df_master['Res_Mom'] = res_mom
            df_master['Fund_Mom'] = fund_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            
            if 'sector' in fundamentals.columns:
                df_master['Sector'] = fundamentals['sector']
            
            df_master.dropna(thresh=2, inplace=True)

            # Z-Score Robusto
            cols_to_norm = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
            norm_cols = []
            for c in cols_to_norm:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    norm_cols.append(new_col)
            
            weights_dict = {
                'Res_Mom_Z': w_rm,
                'Fund_Mom_Z': w_fm,
                'Value_Z': w_val,
                'Quality_Z': w_qual
            }
            
            final_df = build_composite_score(df_master, weights_dict)
            status.update(label="Concluído!", state="complete", expanded=False)

        # --- OUTPUTS ---
        
        tab1, tab2, tab3 = st.tabs(["🏆 Ranking & Seleção", "📈 Backtest (In-Sample)", "🔍 Detalhes dos Fatores"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Top Picks")
                show_cols = ['Composite_Score', 'Sector'] + norm_cols
                st.dataframe(
                    final_df[show_cols].style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']),
                    height=400,
                    use_container_width=True
                )
            
            with col2:
                st.subheader("Alocação Sugerida")
                weights = construct_portfolio(final_df, prices, top_n, target_vol)
                
                if not weights.empty:
                    w_df = weights.to_frame(name="Peso")
                    w_df["Peso"] = w_df["Peso"].map("{:.1%}".format)
                    st.table(w_df)
                    
                    fig_pie = px.pie(values=weights.values, names=weights.index, title="Distribuição")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Performance Recente (Simulação)")
            st.info("Nota: Este backtest simula como a carteira SELECIONADA HOJE teria performado nos últimos 12 meses (In-Sample Analysis).")
            
            if not weights.empty:
                curve = run_backtest(weights, prices)
                
                if not curve.empty:
                    tot_ret = curve['Strategy'].iloc[-1] - 1
                    vol = curve['Strategy'].pct_change().std() * (252**0.5)
                    sharpe = tot_ret / vol if vol > 0 else 0
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Retorno Total", f"{tot_ret:.2%}")
                    m2.metric("Volatilidade Anual", f"{vol:.2%}")
                    m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    fig = px.line(curve, title="Equity Curve: Estratégia vs IBOV11.SA")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Dados de preço insuficientes para gerar o gráfico.")

        with tab3:
            st.subheader("Correlação entre Fatores")
            if norm_cols:
                corr = final_df[norm_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Mapa de Calor de Correlação")
                st.plotly_chart(fig_corr)
            
            st.subheader("Dados Fundamentais Brutos")
            st.dataframe(fundamentals)

if __name__ == "__main__":
    main()
