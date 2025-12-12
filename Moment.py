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
# Este é um subconjunto robusto de ações brasileiras para garantir pares setoriais.
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
        
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data.dropna(how='all')
    except Exception as e:
        st.error(f"Erro ao baixar preços: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=3600*24)
def fetch_fundamentals(tickers: list) -> pd.DataFrame:
    """Busca snapshots fundamentais atuais de um universo amplo para normalização."""
    
    # Combina a lista do usuário com o universo amplo para garantir a base de comparação setorial
    all_tickers = list(set(tickers) | set(BROAD_UNIVERSE))
    clean_tickers = [t for t in all_tickers if t != 'BOVA11.SA']
    
    data = []
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    # Apenas busca os dados para os tickers necessários
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
            temp_df[m] = s
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor: Inverso de P/E e P/B. Ponderação 50/50."""
    scores = pd.DataFrame(index=fund_df.index)
    # EP (Earnings Yield = 1/P/E)
    if 'forwardPE' in fund_df.columns: 
        scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
    # BP (Book to Price = 1/P/B)
    if 'priceToBook' in fund_df.columns: 
        scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
        
    return scores.mean(axis=1).rename("Value_Score")

def compute_quality_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Qualidade: ROE, Margem e Alavancagem. Ponderação 1/3 para cada."""
    scores = pd.DataFrame(index=fund_df.index)
    # ROE e Margem
    if 'returnOnEquity' in fund_df.columns: scores['ROE'] = fund_df['returnOnEquity']
    if 'profitMargins' in fund_df.columns: scores['PM'] = fund_df['profitMargins']
    # Inverso do Dívida/Patrimônio (Alavancagem - menor é melhor, por isso o sinal negativo)
    if 'debtToEquity' in fund_df.columns: 
        # Trata DTE < 0 como 0 para evitar inversão de sinal no score final.
        safe_dte = np.where(fund_df['debtToEquity'] > 0, fund_df['debtToEquity'], 0)
        scores['DE_Inv'] = -1 * safe_dte
        
    return scores.mean(axis=1).rename("Quality_Score")

# ==============================================================================
# MÓDULO 3: SCORING & NORMALIZAÇÃO
# ==============================================================================

def robust_zscore(series: pd.Series) -> pd.Series:
    """Z-Score Robusto (Usa Mediana e MAD para robustez contra outliers)."""
    # Remove infinitos e NaNs antes do cálculo
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty: return pd.Series(np.nan, index=series.index)
        
    median = series.median()
    mad = (series - median).abs().median()
    
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
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
            
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 4: PORTFOLIO & BACKTEST (NORMALIZAÇÃO FORÇADA)
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    """Define pesos do portfólio (Equal Weight ou Risco Inverso, sempre somando 100%)."""
    # Garante que top_n não seja maior que o número de ativos no ranking
    top_n = min(top_n, len(ranked_df))
    
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    if vol_target is not None:
        # Ponderação por Risco Inverso 
        
        # 1. Calcular volatilidade histórica (3 meses / 63 dias)
        recent_rets = prices[selected].pct_change().tail(63)
        vols = recent_rets.std() * (252**0.5)
        vols[vols == 0] = 1e-6 # Evita divisão por zero
        
        # 2. Calcular Pesos de Risco Inverso
        raw_weights_inv = 1 / vols
        
        # 3. FORÇA A NORMALIZAÇÃO para 100%
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
    
    # Garante que apenas os tickers do usuário sejam usados no ranking final
    user_tickers = [t.strip().upper() for t in ticker_input.split(',') if t.strip()]

    st.sidebar.header("2. Pesos dos Fatores (Alpha)")
    w_rm = st.sidebar.slider("Residual Momentum", 0.0, 1.0, 0.40)
    w_fm = st.sidebar.slider("Fundamental Momentum", 0.0, 1.0, 0.20)
    w_val = st.sidebar.slider("Value", 0.0, 1.0, 0.20)
    w_qual = st.sidebar.slider("Quality", 0.0, 1.0, 0.20)

    st.sidebar.header("3. Construção de Portfólio (Risco)")
    top_n = st.sidebar.number_input("Número de Ativos (Top N)", 1, 20, 10)
    
    # Ponderação por Risco Inverso
    use_vol_target = st.sidebar.checkbox("Usar Ponderação por Risco Inverso?", True)
    target_vol = st.sidebar.slider("Volatilidade Alvo (Apenas para referência)", 0.05, 0.30, 0.15) if use_vol_target else None
    
    run_btn = st.sidebar.button("🚀 Rodar Análise", type="primary")

    # --- MAIN LOGIC ---
    if run_btn:
        if not user_tickers:
            st.error("Por favor, insira pelo menos um ticker.")
            return

        with st.status("Executando Pipeline Quant...", expanded=True) as status:
            
            # 1. Dados: Fetch de preços (apenas user_tickers) e fundamentos (broad + user)
            st.write("📥 Baixando dados de mercado e fundamentais (Comparação Ampla)...")
            end_date = datetime.now()
            start_date = end_date - timedelta(days=730)
            
            # Preços só precisam dos tickers do usuário + BOVA11
            prices = fetch_price_data(user_tickers, start_date, end_date) 
            
            # Fundamentos usam o universo ampliado para normalização
            fundamentals_broad = fetch_fundamentals(user_tickers)
            
            if prices.empty or fundamentals_broad.empty:
                st.error("Não foi possível obter dados suficientes.")
                status.update(label="Erro!", state="error")
                return

            # 2. Cálculos: Feitos sobre o universo amplo de fundamentos
            st.write("🧮 Calculando fatores multifatoriais...")
            res_mom = compute_residual_momentum(prices) # Residual Mom: Feito apenas no user_tickers
            fund_mom = compute_fundamental_momentum(fundamentals_broad)
            val_score = compute_value_score(fundamentals_broad)
            qual_score = compute_quality_score(fundamentals_broad)

            # 3. Consolidação e Normalização (Ajuste para Comparação Externa)
            st.write("⚖️ Normalizando e Ranking (Value e Quality setorialmente vs. Mercado Amplo)...")
            
            # DataFrame mestre com todos os ativos do fetch
            df_master = pd.DataFrame(index=fundamentals_broad.index)
            df_master['Res_Mom'] = res_mom
            df_master['Fund_Mom'] = fund_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            
            if 'sector' in fundamentals_broad.columns:
                df_master['Sector'] = fundamentals_broad['sector']
            
            # Filtra o master para incluir apenas os tickers do usuário para o ranking final
            df_master_filtered = df_master.loc[df_master.index.intersection(user_tickers)].copy()

            # Z-Score Robusto sobre o universo amplo (inclusive dos tickers não filtrados)
            cols_to_norm_sectorial = ['Value', 'Quality']
            cols_to_norm_global = ['Res_Mom', 'Fund_Mom'] 

            norm_cols = []
            
            # Normalização Setorial (Value e Quality): usa o universe amplo para o cálculo
            if 'Sector' in df_master.columns:
                for c in cols_to_norm_sectorial:
                    if c in df_master.columns:
                        new_col = f"{c}_Z"
                        # Aplica o Z-Score robusto AGRUPADO pelo setor no universo amplo
                        df_master[new_col] = df_master.groupby('Sector')[c].transform(robust_zscore)
                        
                        # Transfere APENAS os Z-Scores calculados para o DF filtrado
                        if new_col in df_master.columns:
                             df_master_filtered[new_col] = df_master[new_col]
                             norm_cols.append(new_col)
            
            # Normalização Global (Momentum): usa o universo amplo para o cálculo
            for c in cols_to_norm_global:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    
                    # Transfere APENAS os Z-Scores calculados para o DF filtrado
                    if new_col in df_master.columns:
                        df_master_filtered[new_col] = df_master[new_col]
                        norm_cols.append(new_col)
            
            # Remove ativos que não puderam ser pontuados (NaN após normalização)
            df_master_filtered.dropna(subset=[f'{c}_Z' for c in cols_to_norm_sectorial if f'{c}_Z' in df_master_filtered.columns] + [f'{c}_Z' for c in cols_to_norm_global if f'{c}_Z' in df_master_filtered.columns], how='all', inplace=True)

            weights_dict = {
                'Res_Mom_Z': w_rm, 'Fund_Mom_Z': w_fm, 
                'Value_Z': w_val, 'Quality_Z': w_qual
            }
            
            # Calcula o Score Composto final APENAS nos tickers do usuário
            final_df = build_composite_score(df_master_filtered, weights_dict)
            
            st.write("⚖️ Calculando alocação e backtest...")
            weights = construct_portfolio(final_df, prices, top_n, target_vol)
            
            # Junta os dados brutos com o score final para a aba de detalhes
            fundamentals_final = fundamentals_broad.loc[final_df.index].join(final_df[norm_cols + ['Composite_Score', 'Sector']])
            

            status.update(label="Concluído!", state="complete", expanded=False)

        # --- OUTPUTS ---
        
        tab1, tab2, tab3, tab4 = st.tabs(["🏆 Ranking & Seleção", "📈 Backtest (In-Sample)", "🔍 Dados Fundamentais", "📜 Justificativa da Seleção"])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("Top Picks (Selecionados pelo Score)")
                st.markdown(f"**Nota:** Os scores de Valor e Qualidade são calculados em relação a um universo amplo de mais de {len(BROAD_UNIVERSE)} ativos, garantindo maior efetividade na comparação setorial.")
                
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
                    
                    st.metric("Soma da Alocação", f"{total_sum:.2%}")
                    
                    w_df["Peso"] = w_df["Peso"].map("{:.2%}".format)
                    st.table(w_df)
                    
                    fig_pie = px.pie(values=weights.values, names=weights.index, title="Distribuição")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Performance Recente (Simulação de 1 Ano)")
            st.info(f"Portfólio de {len(weights)} ativos rebalanceado por Risco Inverso (Target Vol é apenas referência).")
            
            if not weights.empty:
                curve = run_backtest(weights, prices)
                
                if not curve.empty and len(curve) > 1:
                    daily_rets = curve.pct_change().dropna()
                    
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
            st.subheader("Dados Fundamentais Brutos (Apenas ativos selecionados)")
            st.info("Valores de Valor (P/L, P/VPA) e Qualidade (ROE, Margem) são normalizados setorialmente contra um universo amplo na aba Ranking.")
            
            show_fund_cols = ['sector', 'forwardPE', 'priceToBook', 'returnOnEquity', 'profitMargins', 'debtToEquity', 'earningsGrowth', 'revenueGrowth']
            st.dataframe(fundamentals_broad.loc[final_df.index, show_fund_cols].style.format("{:.2f}", subset=['forwardPE', 'priceToBook', 'enterpriseToEbitda', 'debtToEquity']).format("{:.2%}", subset=['returnOnEquity', 'profitMargins', 'earningsGrowth', 'revenueGrowth']))
            
            st.subheader("Correlação entre Fatores (Normalizados)")
            if norm_cols:
                corr = final_df[norm_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Mapa de Calor de Correlação")
                st.plotly_chart(fig_corr)

        with tab4:
            st.subheader("Justificativa Detalhada da Seleção (Top Picks)")
            st.markdown("A pontuação de cada ativo é baseada no seu Z-Score em cada fator. **Scores positivos** indicam que o ativo está **melhor** do que a média de seus pares (setorial ou global).")
            
            # Tabela de Score Detalhado
            score_cols = ['Composite_Score', 'Sector'] + norm_cols
            detailed_df = fundamentals_final.sort_values('Composite_Score', ascending=False).head(top_n)[score_cols]
            
            st.dataframe(
                detailed_df.style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']).format("{:.2f}"),
                width='stretch'
            )

            # Detalhamento por Ticker
            st.markdown("---")
            for ticker in detailed_df.index:
                row = detailed_df.loc[ticker]
                st.markdown(f"### 📈 {ticker} - Score: {row['Composite_Score']:.2f}")
                st.markdown(f"**Setor:** {row['Sector']}")
                
                justification = []
                
                # Fator 1: Momentum de Preço
                score = row.get('Res_Mom_Z', 0)
                if score > 0.5:
                    justification.append(f"- **Residual Momentum ({score:.2f}):** Muito acima da média. A ação tem gerado forte **Alpha (retorno não explicado pelo mercado)** recentemente.")
                elif score > 0.0:
                    justification.append(f"- **Residual Momentum ({score:.2f}):** Acima da média. Forte sinal de que o preço superou o benchmark (BOVA11.SA).")
                else:
                    justification.append(f"- **Residual Momentum ({score:.2f}):** Abaixo ou na média. O preço não gerou Alpha significativo recentemente.")

                # Fator 2: Momentum Fundamental
                score = row.get('Fund_Mom_Z', 0)
                if score > 0.5:
                    justification.append(f"- **Fundamental Momentum ({score:.2f}):** Forte crescimento de Lucro/Receita (top do universo de comparação).")
                elif score > 0.0:
                    justification.append(f"- **Fundamental Momentum ({score:.2f}):** Crescimento acima da média do mercado.")
                else:
                    justification.append(f"- **Fundamental Momentum ({score:.2f}):** Crescimento de lucro/receita abaixo da média do mercado.")
                    
                # Fator 3: Valor (Setorial)
                score = row.get('Value_Z', 0)
                if score > 0.5:
                    justification.append(f"- **Value (Valor) ({score:.2f}):** Extremamente barato em relação aos **pares do seu setor** (Altos 1/P/L e 1/P/VPA). Forte desconto.")
                elif score > 0.0:
                    justification.append(f"- **Value (Valor) ({score:.2f}):** Barato em relação à média setorial. Bom indicativo de preço atrativo.")
                else:
                    justification.append(f"- **Value (Valor) ({score:.2f}):** Preço na média ou ligeiramente caro em relação aos seus pares setoriais.")
                    
                # Fator 4: Qualidade (Setorial)
                score = row.get('Quality_Z', 0)
                if score > 0.5:
                    justification.append(f"- **Quality (Qualidade) ({score:.2f}):** Alta eficiência (ROE e Margem) e baixa alavancagem em relação aos **pares do seu setor**.")
                elif score > 0.0:
                    justification.append(f"- **Quality (Qualidade) ({score:.2f}):** Qualidade (ROE, Margem, Dívida) acima da média setorial.")
                else:
                    justification.append(f"- **Quality (Qualidade) ({score:.2f}):** Qualidade (ROE, Margem, Dívida) na média ou ligeiramente abaixo da média setorial.")

                st.info("\n".join(justification))


if __name__ == "__main__":
    main()
