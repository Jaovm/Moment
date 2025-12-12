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

# Universo Amplo para Normalização (Comparação Externa)
BROAD_UNIVERSE = [
    'VALE3.SA', 'PETR4.SA', 'ITUB3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA', 
    'RENT3.SA', 'WEGE3.SA', 'B3SA3.SA', 'SUZB3.SA', 'BBAS3.SA', 'PRIO3.SA', 
    'GGBR4.SA', 'CMIG4.SA', 'ELET3.SA', 'ENBR3.SA', 'CSAN3.SA', 'COGN3.SA',
    'RADL3.SA', 'MGLU3.SA', 'GOAU4.SA', 'VIVT3.SA', 'BRFS3.SA', 'HAPV3.SA',
    'MRFG3.SA', 'AZUL4.SA', 'CVCB3.SA', 'PETR3.SA', 'SANB11.SA', 'BPAC11.SA', 
    'TAEE11.SA', 'SBSP3.SA', 'SAPR4.SA', 'EGIE3.SA', 'MDIA3.SA', 'TOTS3.SA', 
    'BBSE3.SA', 'PSSA3.SA', 'FRAS3.SA', 'AGRO3.SA', 'LREN3.SA', 'AMER3.SA' 
]

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
        
        # Correção para yfinance retornando MultiIndex nas colunas
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        # Garante que é um DataFrame mesmo com 1 ticker
        if isinstance(data, pd.Series):
            data = data.to_frame()
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """Busca snapshots fundamentais atuais de um universo amplo para normalização."""
    
    all_tickers = list(set(tickers) | set(BROAD_UNIVERSE))
    # Remove duplicatas e garante BOVA11 fora dos fundamentos
    clean_tickers = [t for t in all_tickers if t != 'BOVA11.SA']
    
    data = []
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            # Tenta pegar info. Se falhar, pula o ticker sem quebrar o app
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            
            # Validação básica: se não tem setor, provavelmente o dado está ruim
            if 'sector' not in info and 'Sector' not in info:
                pass 
                
            data.append({
                'ticker': t,
                'Sector': info.get('sector', info.get('Sector', 'Unknown')), # Padroniza para 'Sector'
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan)
            })
        except Exception:
            pass # Ignora erros de conexão pontuais
            
        if (i + 1) % 5 == 0: # Atualiza a barra menos vezes para performance
            progress_bar.progress(min((i + 1) / total, 1.0))
        
    progress_bar.empty()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data).set_index('ticker')
    # Remove linhas onde tudo é NaN (exceto Sector)
    cols_to_check = [c for c in df.columns if c != 'Sector']
    df = df.dropna(subset=cols_to_check, how='all')
    
    return df

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Calcula Residual Momentum (Alpha) vs BOVA11.SA."""
    if price_df.empty: return pd.Series(dtype=float)
    
    df = price_df.copy()
    # Resample mensal ('ME' é o novo alias para Month End no pandas 2.2+, 'M' para antigos)
    try:
        monthly = df.resample('ME').last()
    except:
        monthly = df.resample('M').last()
        
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
            temp_df[m] = s
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor: Inverso de P/E e P/B. Ponderação 50/50."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'forwardPE' in fund_df.columns: 
        scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
    if 'priceToBook' in fund_df.columns: 
        scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
        
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'returnOnEquity' in fund_df.columns: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df.columns: scores['PM'] = fund_df['profitMargins']
    if 'debtToEquity' in fund_df.columns: 
        safe_dte = np.where(fund_df['debtToEquity'] > 0, fund_df['debtToEquity'], 0)
        scores['DE_Inv'] = -1 * safe_dte
        
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto."""
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty: return pd.Series(np.nan, index=series.index)
        
    median = series.median()
    mad = (series - median).abs().median()
    
    z = pd.Series(np.nan, index=series.index)
    
    if mad == 0: 
        z.loc[series.index] = series - median
    else:
        z.loc[series.index] = (series - median) / (mad * 1.4826)
        
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
    """Define pesos do portfólio."""
    # Filtra para garantir que só temos ativos com preço disponível
    available_tickers = [t for t in ranked_df.index if t in prices.columns]
    selected = ranked_df.loc[available_tickers].head(top_n).index.tolist()
    
    if not selected: return pd.Series()

    if vol_target is not None:
        recent_rets = prices[selected].pct_change().tail(63)
        if recent_rets.empty: return pd.Series(1.0/len(selected), index=selected)
        
        vols = recent_rets.std() * (252**0.5)
        vols = vols.replace(0, 1e-6) # Evita div por zero
        
        raw_weights_inv = 1 / vols
        weights = raw_weights_inv / raw_weights_inv.sum() 
    else:
        weights = pd.Series(1.0/len(selected), index=selected)
        
    return weights

def run_backtest(weights: pd.Series, prices: pd.DataFrame, lookback_days: int = 252):
    """Simula o desempenho."""
    valid_tickers = [t for t in weights.index if t in prices.columns]
    if not valid_tickers: return pd.DataFrame()
        
    subset = prices.tail(lookback_days)
    rets = subset.pct_change().dropna()
    
    if rets.empty: return pd.DataFrame()
    
    port_ret = rets[valid_tickers].dot(weights[valid_tickers].fillna(0))
    BVSP_ret = rets['BOVA11.SA'] if 'BOVA11.SA' in rets.columns else pd.Series(0, index=rets.index)
    
    daily_rets = pd.DataFrame({'Strategy': port_ret, 'BOVA11.SA': BVSP_ret})
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
    
    user_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores (Alpha)")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_fm = st.sidebar.slider("Fundamental Momentum", 0.0, 1.0, 0.20)
    w_val = st.sidebar.slider("Value", 0.0, 1.0, 0.20)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Construção de Portfólio (Risco)")
    top_n = st.sidebar.number_input("Número de Ativos (Top N)", 1, 20, 10)
    use_vol_target = st.sidebar.checkbox("Usar Ponderação por Risco Inverso?", True)
    target_vol = st.sidebar.slider("Volatilidade Alvo (Apenas para referência)", 0.05, 0.30, 0.15) if use_vol_target else None
    
    run_btn = st.sidebar.button("🚀 Rodar Análise", type="primary")

    # --- MAIN LOGIC ---
    if run_btn:
        if not user_tickers:
            st.error("Por favor, insira pelo menos um ticker.")
            return

        with st.status("Executando Pipeline Quant...", expanded=True) as status:
            
            st.write("📥 Baixando dados de mercado e fundamentais (Comparação Ampla)...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            prices = fetch_price_data(user_tickers, start_date, end_date) 
            fundamentals_broad = fetch_fundamentals(user_tickers)
            
            if prices.empty or fundamentals_broad.empty:
                st.error("Não foi possível obter dados suficientes. Verifique os tickers.")
                status.update(label="Erro!", state="error")
                return

            st.write("🧮 Calculando fatores multifatoriais...")
            res_mom = compute_residual_momentum(prices)
            fund_mom = compute_fundamental_momentum(fundamentals_broad)
            val_score = compute_value_score(fundamentals_broad)
            qual_score = compute_quality_score(fundamentals_broad)

            st.write("⚖️ Normalizando e Ranking (Setorial vs. Mercado Amplo)...")
            
            df_master = pd.DataFrame(index=fundamentals_broad.index)
            df_master['Res_Mom'] = res_mom
            df_master['Fund_Mom'] = fund_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            
            # Garante o nome correto da coluna de setor (Case sensitive fix)
            if 'Sector' in fundamentals_broad.columns:
                df_master['Sector'] = fundamentals_broad['Sector']
            
            # Filtra apenas tickers do usuário que existem nos dados baixados
            valid_user_tickers = [t for t in user_tickers if t in df_master.index]
            df_master_filtered = df_master.loc[valid_user_tickers].copy()
            
            if df_master_filtered.empty:
                st.error("Nenhum ticker válido encontrado nos dados fundamentais.")
                status.update(label="Erro!", state="error")
                return

            cols_to_norm_sectorial = ['Value', 'Quality']
            cols_to_norm_global = ['Res_Mom', 'Fund_Mom'] 
            norm_cols = []
            
            # Normalização
            if 'Sector' in df_master.columns:
                for c in cols_to_norm_sectorial:
                    if c in df_master.columns:
                        new_col = f"{c}_Z"
                        # Transformação ocorre no DF amplo, depois mapeamos para o filtrado
                        df_master[new_col] = df_master.groupby('Sector')[c].transform(robust_zscore)
                        
                        # Atribuição segura usando .loc
                        df_master_filtered[new_col] = df_master.loc[df_master_filtered.index, new_col]
                        norm_cols.append(new_col)
            
            for c in cols_to_norm_global:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    
                    df_master_filtered[new_col] = df_master.loc[df_master_filtered.index, new_col]
                    norm_cols.append(new_col)
            
            # Limpeza de NaNs nos scores
            final_cols = [f'{c}_Z' for c in cols_to_norm_sectorial if f'{c}_Z' in df_master_filtered] + \
                         [f'{c}_Z' for c in cols_to_norm_global if f'{c}_Z' in df_master_filtered]
                         
            df_master_filtered.dropna(subset=final_cols, how='all', inplace=True)

            weights_dict = {
                'Res_Mom_Z': w_rm, 'Fund_Mom_Z': w_fm, 
                'Value_Z': w_val, 'Quality_Z': w_qual
            }
            
            final_df = build_composite_score(df_master_filtered, weights_dict)
            
            st.write("⚖️ Calculando alocação e backtest...")
            weights = construct_portfolio(final_df, prices, top_n, target_vol)
            
            # --- CORREÇÃO DO JOIN (Evita erro de sobreposição de colunas) ---
            # Selecionamos apenas as colunas CALCULADAS do final_df para juntar com os dados brutos
            cols_to_merge = [c for c in final_df.columns if c not in fundamentals_broad.columns]
            
            # Merge seguro usando índices
            fundamentals_final = pd.merge(
                fundamentals_broad.loc[final_df.index],
                final_df[cols_to_merge],
                left_index=True,
                right_index=True,
                how='inner'
            )

            status.update(label="Concluído!", state="complete", expanded=False)

        # --- OUTPUTS ---
        
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking & Seleção", "📈 Backtest (In-Sample)", "🔍 Dados Fundamentais", "📜 Justificativa da Seleção"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Top Picks")
                st.markdown(f"**Nota:** Comparação feita contra universo de {len(fundamentals_broad)} ativos.")
                
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
                    st.metric("Soma da Alocação", f"{weights.sum():.2%}")
                    w_df["Peso"] = w_df["Peso"].map("{:.2%}".format)
                    st.table(w_df)
                    fig_pie = px.pie(values=weights.values, names=weights.index, title="Distribuição")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Simulação (Backtest)")
            if not weights.empty:
                curve = run_backtest(weights, prices)
                if not curve.empty and len(curve) > 1:
                    tot_ret = curve['Strategy'].iloc[-1] - 1
                    daily_rets = curve.pct_change().dropna()
                    vol = daily_rets['Strategy'].std() * (252**0.5)
                    sharpe = tot_ret / vol if vol > 0 else 0
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Retorno Total", f"{tot_ret:.2%}")
                    m2.metric("Volatilidade", f"{vol:.2%}")
                    m3.metric("Sharpe", f"{sharpe:.2f}")
                    
                    fig = px.line(curve, title="Equity Curve")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Dados insuficientes para backtest.")
            else:
                st.warning("Nenhum ativo selecionado.")

        with tab3:
            st.subheader("Dados Fundamentais Brutos")
            st.dataframe(fundamentals_broad.loc[final_df.index])
            
            if norm_cols:
                st.subheader("Correlação (Fatores Normalizados)")
                corr = final_df[norm_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr)

        with tab4:
            st.subheader("Justificativa Detalhada")
            
            # Prepara dados para exibição (ordena pelo score)
            detailed_df = fundamentals_final.sort_values('Composite_Score', ascending=False).head(top_n)
            
            for ticker in detailed_df.index:
                row = detailed_df.loc[ticker]
                st.markdown(f"### 📈 {ticker} - Score: {row.get('Composite_Score', 0):.2f}")
                
                # Check seguro para Sector
                sector_val = row.get('Sector', 'N/A')
                st.markdown(f"**Setor:** {sector_val}")
                
                justification = []
                
                # Helper para criar texto
                def analyze_factor(factor_name, label_name, high_desc, low_desc):
                    val = row.get(factor_name, 0)
                    if pd.isna(val): return
                    if val > 0.5: justification.append(f"- **{label_name} ({val:.2f}):** {high_desc}")
                    elif val > 0.0: justification.append(f"- **{label_name} ({val:.2f}):** Acima da média.")
                    else: justification.append(f"- **{label_name} ({val:.2f}):** {low_desc}")

                analyze_factor('Res_Mom_Z', 'Momentum Preço', 'Forte tendência de alta vs Ibov.', 'Preço sem força relativa.')
                analyze_factor('Fund_Mom_Z', 'Momentum Fundamental', 'Lucros/Receita acelerando forte.', 'Crescimento estagnado/negativo.')
                analyze_factor('Value_Z', 'Valor (vs Setor)', 'Muito descontado (P/L e P/VP baixos).', 'Preço justo ou caro vs pares.')
                analyze_factor('Quality_Z', 'Qualidade (vs Setor)', 'Alta eficiência (ROE) e solidez.', 'Qualidade na média ou abaixo.')

                st.info("\n".join(justification) if justification else "Sem dados suficientes para justificativa.")

if __name__ == "__main__":
    main()
