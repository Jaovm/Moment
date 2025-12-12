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
    """Busca histórico de preços ajustados, garantindo o benchmark BOVA11.SA."""
    t_list = list(tickers)
    if 'BOVA11.SA' not in t_list:
        t_list.append('BOVA11.SA')
    
    try:
        data = yf.download(
            t_list, 
            start=start_date, 
            end=end_date, 
            progress=False,
            auto_adjust=False
        )['Adj Close']
        
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
    clean_tickers = [t for t in tickers if t != 'BOVA11.SA']
    
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            info = yf.Ticker(t).info
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
    """Calcula Residual Momentum (Alpha) vs BOVA11.SA."""
    df = price_df.copy()
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    window = lookback + skip
    
    for ticker in rets.columns:
        if ticker == 'BOVA11.SA': continue
        
        y = rets[ticker].tail(window)
        x = market.tail(window)
        
        if len(y) < window: continue
            
        try:
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            resid = model.resid[:-skip]
            
            if np.std(resid) == 0 or len(resid) < 2:
                scores[ticker] = 0
            else:
                scores[ticker] = np.sum(resid) / np.std(resid)
        except:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Residual_Momentum')

def compute_fundamental_momentum(fund_df: pd.DataFrame) -> pd.Series:
    """Z-Score combinado de crescimento de Receita e Lucro."""
    metrics = ['earningsGrowth', 'revenueGrowth']
    temp_df = pd.DataFrame(index=fund_df.index)
    for m in metrics:
        if m in fund_df.columns:
            s = fund_df[m].fillna(fund_df[m].median())
            # Não faz Z-score aqui. Apenas soma os valores para serem normalizados depois.
            temp_df[m] = s
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor: Inverso de P/E e P/B. Ponderação 50/50."""
    scores = pd.DataFrame(index=fund_df.index)
    # EP (Earnings Yield)
    if 'forwardPE' in fund_df.columns: 
        scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
    # BP (Book to Price)
    if 'priceToBook' in fund_df.columns: 
        scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
        
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem. Ponderação 1/3 para cada."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df.columns: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df.columns: scores['PM'] = fund_df['profitMargins']
    # Inverso do Dívida/Patrimônio
    if 'debtToEquity' in fund_df.columns: 
        scores['DE_Inv'] = np.where(fund_df['debtToEquity'] > 0, -1 * fund_df['debtToEquity'], 0)
        
    # A Média é feita sobre os scores (que serão normalizados depois)
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto (Usa Mediana e MAD para robustez contra outliers)."""
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty: return pd.Series(np.nan, index=series.index)
        
    median = series.median()
    mad = (series - median).abs().median()
    
    # Adiciona índice para evitar erro de alinhamento no .transform()
    z = pd.Series(np.nan, index=series.index)
    
    if mad == 0: 
        z.loc[series.index] = series - median
    else:
        # 1.4826 é o fator de consistência para a Distribuição Normal
        z.loc[series.index] = (series - median) / (mad * 1.4826)
        
    return z.clip(-3, 3)

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Calcula score final ponderado."""
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            # Multiplica o fator normalizado pelo peso. Usa fillna(0) antes de somar.
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
            
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 4: PORTFOLIO & BACKTEST (NORMALIZAÇÃO FORÇADA)
# ==============================================================================

# (O restante do código, Módulo 4 e APP Principal, não precisa de grandes alterações,
# mas estou incluindo o MAIN com a lógica de normalização ajustada)

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    """Define pesos do portfólio (Equal Weight ou Risco Inverso, sempre somando 100%)."""
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    if vol_target is not None:
        # Ponderação por Risco Inverso (sem dimensionamento pelo alvo, mas com normalização)
        
        # 1. Calcular volatilidade histórica (3 meses / 63 dias)
        recent_rets = prices[selected].pct_change().tail(63)
        vols = recent_rets.std() * (252**0.5)
        vols[vols == 0] = 1e-6 # Evita divisão por zero
        
        # 2. Calcular Pesos de Risco Inverso
        raw_weights_inv = 1 / vols
        
        # 3. FORÇA A NORMALIZAÇÃO para 100% (Desliga o dimensionamento do Vol Target)
        weights = raw_weights_inv / raw_weights_inv.sum() 
            
    else:
        # Pesos Iguais (Equal Weight)
        weights = pd.Series(1.0/len(selected), index=selected)
        
    return weights

def run_backtest(weights: pd.Series, prices: pd.DataFrame, lookback_days: int = 252):
    """Simula o desempenho do portfólio selecionado."""
    valid_tickers = [t for t in weights.index if t in prices.columns]
    if not valid_tickers: return pd.DataFrame()
        
    # Recorta o período
    subset = prices.tail(lookback_days)
    rets = subset.pct_change().dropna()
    
    # 1. Retorno do Portfólio (Usa os pesos normalizados a 100%)
    port_ret = rets[valid_tickers].dot(weights[valid_tickers].fillna(0))
    
    # 2. Retorno do Benchmark
    BVSP_ret = rets['BOVA11.SA'] if 'BOVA11.SA' in rets.columns else pd.Series(0, index=rets.index)
    
    # Cria DataFrame de retornos diários
    daily_rets = pd.DataFrame({'Strategy': port_ret, 'BOVA11.SA': BVSP_ret})
    
    # Retorno Cumulativo (Começa em 1.0)
    cumulative = (1 + daily_rets).cumprod()
    return cumulative.dropna()

# ==============================================================================
# APP PRINCIPAL (STREAMLIT UI)
# ==============================================================================

def main():
    st.title("🧪 Quant Factor Lab: Multi-Strategy Engine")
    st.markdown("Otimização de carteira Long-Only baseada em fatores e risco.")

    # --- SIDEBAR ---
    st.sidebar.header("1. Universo e Dados (BOVESPA)")
    default_univ = "ITUB3.SA, TOTS3.SA, MDIA3.SA, TAEE3.SA, BBSE3.SA, WEGE3.SA, PSSA3.SA, EGIE3.SA, B3SA3.SA, VIVT3.SA, AGRO3.SA, PRIO3.SA, BBAS3.SA, BPAC11.SA, SBSP3.SA, SAPR4.SA, CMIG3.SA, UNIP6.SA, FRAS3.SA"
    ticker_input = st.sidebar.text_area("Tickers (Separados por vírgula)", default_univ, height=100)
    tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores (Alpha)")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_fm = st.sidebar.slider("Fundamental Momentum", 0.0, 1.0, 0.20)
    w_val = st.sidebar.slider("Value", 0.0, 1.0, 0.20)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Construção de Portfólio (Risco)")
    top_n = st.sidebar.number_input("Número de Ativos (Top N)", 1, 20, 5)
    
    # O checkbox agora ativa a ponderação de Risco Inverso, mas a normalização é forçada
    use_vol_target = st.sidebar.checkbox("Usar Ponderação por Risco Inverso?", True)
    
    # target_vol é mantido para manter a interface, mas não afeta a escala total
    target_vol = st.sidebar.slider("Volatilidade Alvo (Apenas para referência)", 0.05, 0.30, 0.15) if use_vol_target else None
    
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

            # 3. Consolidação e Normalização Setorial (Ajuste principal)
            st.write("⚖️ Normalizando e Ranking (Value e Quality setorialmente)...")
            
            # Alinha o DataFrame mestre com os fundamentos que contêm o setor
            df_master = pd.DataFrame(index=fundamentals.index)
            df_master['Res_Mom'] = res_mom
            df_master['Fund_Mom'] = fund_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            
            if 'sector' in fundamentals.columns:
                df_master['Sector'] = fundamentals['sector']
            
            # Remove linhas que não têm pelo menos 2 fatores calculados
            df_master.dropna(thresh=2, inplace=True) 

            # Define fatores para normalização setorial e global
            cols_to_norm_sectorial = ['Value', 'Quality']
            cols_to_norm_global = ['Res_Mom', 'Fund_Mom'] 

            norm_cols = []
            
            # 3.1 Normalização Setorial (Value e Quality)
            # A ação é comparada apenas com seus pares de setor.
            if 'Sector' in df_master.columns:
                for c in cols_to_norm_sectorial:
                    if c in df_master.columns:
                        new_col = f"{c}_Z"
                        # Aplica o robust_zscore agrupado por 'Sector'
                        df_master[new_col] = df_master.groupby('Sector')[c].transform(robust_zscore)
                        norm_cols.append(new_col)
            
            # 3.2 Normalização Global (Momentum)
            # A ação é comparada a todas as outras no universo.
            for c in cols_to_norm_global:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    norm_cols.append(new_col)
            
            # Remove ativos onde a normalização resultou em NaN (devido a setores com poucas amostras)
            df_master.dropna(subset=[f'{c}_Z' for c in cols_to_norm_sectorial if f'{c}_Z' in df_master.columns], inplace=True)

            weights_dict = {
                'Res_Mom_Z': w_rm, 'Fund_Mom_Z': w_fm, 
                'Value_Z': w_val, 'Quality_Z': w_qual
            }
            
            final_df = build_composite_score(df_master, weights_dict)
            
            st.write("⚖️ Calculando alocação e backtest...")
            weights = construct_portfolio(final_df, prices, top_n, target_vol)

            status.update(label="Concluído!", state="complete", expanded=False)

        # --- OUTPUTS ---
        
        tab1, tab2, tab3 = st.tabs(["🏆 Ranking & Seleção", "📈 Backtest (In-Sample)", "🔍 Detalhes dos Fatores"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Top Picks (Selecionados pelo Score)")
                show_cols = ['Composite_Score', 'Sector'] + norm_cols
                st.dataframe(
                    final_df[show_cols].head(top_n).style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']),
                    height=400,
                    width='stretch'
                )
            
            with col2:
                st.subheader("Alocação Sugerida")
                if not weights.empty:
                    w_df = weights.to_frame(name="Peso")
                    total_sum = weights.sum()
                    
                    # A Soma Total deve ser sempre 100% agora
                    st.metric("Soma da Alocação", f"{total_sum:.2%}")
                    
                    w_df["Peso"] = w_df["Peso"].map("{:.2%}".format)
                    st.table(w_df)
                    
                    fig_pie = px.pie(values=weights.values, names=weights.index, title="Distribuição")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Performance Recente (Simulação de 1 Ano)")
            st.info("A alocação está sempre em 100% (Risco Inverso Normalizado).")
            
            if not weights.empty:
                curve = run_backtest(weights, prices)
                
                if not curve.empty and len(curve) > 1:
                    # Cálculo de Métricas
                    daily_rets = curve.pct_change().dropna()
                    
                    # Cálculo das métricas anuais
                    tot_ret = curve['Strategy'].iloc[-1] - 1
                    vol = daily_rets['Strategy'].std() * (252**0.5)
                    ret_bench = curve['BOVA11.SA'].iloc[-1] - 1 if 'BOVA11.SA' in curve.columns else np.nan
                    
                    sharpe = tot_ret / vol if vol > 0 else 0
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Retorno Total (Estratégia)", f"{tot_ret:.2%}")
                    m2.metric("Volatilidade Anual", f"{vol:.2%}")
                    m3.metric("Sharpe Ratio (Anual)", f"{sharpe:.2f}")
                    if not np.isnan(ret_bench):
                         m4.metric("Retorno Benchmark (BOVA11.SA)", f"{ret_bench:.2%}")
                    
                    fig = px.line(curve, title="Equity Curve: Estratégia vs BOVA11.SA")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Dados insuficientes para calcular o backtest no período.")
            else:
                st.warning("Nenhum ativo selecionado.")

        with tab3:
            st.subheader("Correlação entre Fatores (Normalizados)")
            if norm_cols:
                corr = final_df[norm_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Mapa de Calor de Correlação")
                st.plotly_chart(fig_corr)
            
            st.subheader("Dados Fundamentais Brutos")
            st.dataframe(fundamentals)

if __name__ == "__main__":
    main()
