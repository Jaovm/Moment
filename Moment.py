import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
from datetime import datetime, timedelta
import calendar

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
            temp_df[m] = (s - s.mean()) / s.std()
    return temp_df.mean(axis=1).rename("Fundamental_Momentum")

def compute_value_score(fund_df: pd.DataFrame) -> pd.Series:
    """Score de Valor: Inverso de P/E e P/B."""
    scores = pd.DataFrame(index=fund_df.index)
    if 'forwardPE' in fund_df: scores['EP'] = np.where(fund_df['forwardPE'] > 0, 1/fund_df['forwardPE'], 0)
    if 'priceToBook' in fund_df: scores['BP'] = np.where(fund_df['priceToBook'] > 0, 1/fund_df['priceToBook'], 0)
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
    """Z-Score Robusto."""
    series = series.replace([np.inf, -np.inf], np.nan)
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0: return series - median
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
    """Define pesos do portfólio (Equal Weight ou Risco Inverso, sempre somando 100%)."""
    selected = ranked_df.head(top_n).index.tolist()
    if not selected: return pd.Series()

    if vol_target is not None:
        # Ponderação por Risco Inverso (Normalizada para 100%)
        
        recent_rets = prices[selected].pct_change().tail(63)
        vols = recent_rets.std() * (252**0.5)
        vols[vols == 0] = 1e-6 
        
        raw_weights_inv = 1 / vols
        
        # FORÇA A NORMALIZAÇÃO para 100% 
        weights = raw_weights_inv / raw_weights_inv.sum() 
            
    else:
        # Pesos Iguais (Equal Weight)
        weights = pd.Series(1.0/len(selected), index=selected)
        
    return weights

def run_backtest(weights: pd.Series, prices: pd.DataFrame, lookback_days: int = 252):
    """
    Simula o desempenho do portfólio selecionado e do Benchmark (aporte inicial).
    Retorna a Curva de Equity.
    """
    subset = prices.tail(lookback_days)
    rets = subset.pct_change().dropna()
    
    if 'BOVA11.SA' in rets.columns:
        BVSP_ret = rets['BOVA11.SA']
    else:
        BVSP_ret = pd.Series(0, index=rets.index)
    
    valid_tickers = [t for t in weights.index if t in prices.columns]
    
    if valid_tickers:
        port_ret = rets[valid_tickers].dot(weights[valid_tickers].fillna(0))
    else:
        port_ret = pd.Series(0, index=rets.index)
        
    daily_rets = pd.DataFrame({'Strategy': port_ret, 'BOVA11.SA': BVSP_ret})
    cumulative = (1 + daily_rets).cumprod()
    return cumulative.dropna()

def run_dca_backtest(weights: pd.Series, prices: pd.DataFrame, monthly_contribution: float = 1000, lookback_days: int = 252):
    """
    Simula um Backtest com Aportes Mensais (DCA) nos últimos 252 dias úteis,
    usando pesos estáticos e aportando no último dia útil do mês.
    """
    valid_tickers = [t for t in weights.index if t in prices.columns]
    if not valid_tickers or 'BOVA11.SA' not in prices.columns:
        return pd.DataFrame(), pd.DataFrame()

    # 1. Definir o período de análise (últimos 252 dias úteis)
    if len(prices) < lookback_days:
        st.warning(f"Dados insuficientes. Usando {len(prices)} dias.")
        subset_prices = prices
    else:
        subset_prices = prices.tail(lookback_days)
        
    end_date = subset_prices.index[-1]
    start_date = subset_prices.index[0]

    # 2. Definir as datas de rebalanceamento (último dia útil do mês no período)
    # Gera datas de fim de mês no período, e filtra para garantir que são dias de negociação
    monthly_dates_approx = pd.date_range(start=start_date, end=end_date, freq='BM')
    
    # Mapeia a data de rebalanceamento para o último dia útil de negociação conhecido
    dates = []
    
    for date in monthly_dates_approx:
        # Encontra o último dia útil de negociação <= à data de fim de mês
        valid_dates = subset_prices.index[subset_prices.index <= date]
        if not valid_dates.empty:
            last_trading_day = valid_dates.max()
            if last_trading_day not in dates: # Evita duplicatas se BM for dia de negociação
                dates.append(last_trading_day)

    # Se a última data não for o fim do período (hoje), adiciona o dia final para valorização
    if dates and dates[-1] < end_date:
        dates.append(end_date)
    elif not dates and len(subset_prices) > 0: # Caso extremo, usa o início e fim
        dates = [start_date, end_date]
        
    # As datas de aporte/rebalanceamento serão todos os elementos de 'dates', exceto o último (que é só a valorização final)
    
    
    # 3. Inicializar variáveis
    capital_strategy = pd.Series(0.0, index=subset_prices.index)
    capital_benchmark = pd.Series(0.0, index=subset_prices.index)
    
    # Histórico de transações para a tabela
    transactions_history = []
    
    last_idx = subset_prices.index[0]

    # Loop principal de DCA
    for i in range(len(dates)):
        rebal_date = dates[i]
        
        # A. Valorização do Capital (se houver período anterior)
        if i > 0:
            # Ponto de partida para a valorização é o valor após o aporte anterior
            start_value_strat = capital_strategy.loc[last_idx]
            start_value_bench = capital_benchmark.loc[last_idx]
            
            # Cálculo da valorização (multiplica o valor após o aporte pelos retornos)
            rets_period = subset_prices.loc[last_idx:rebal_date].pct_change().dropna()
            
            # Valorização da Estratégia
            if not rets_period.empty:
                # Usa os retornos do período ponderados pelos pesos estáticos
                valorization_factor_strat = (1 + rets_period.dot(weights[valid_tickers].fillna(0))).prod()
                capital_strategy.loc[rebal_date] = start_value_strat * valorization_factor_strat
            else:
                 capital_strategy.loc[rebal_date] = start_value_strat

            # Valorização do Benchmark
            if 'BOVA11.SA' in rets_period.columns:
                valorization_factor_bench = (1 + rets_period['BOVA11.SA']).prod()
                capital_benchmark.loc[rebal_date] = start_value_bench * valorization_factor_bench
            else:
                capital_benchmark.loc[rebal_date] = start_value_bench
                
            last_idx = rebal_date # O novo ponto de partida
            
        # B. Aplicar o novo Aporte (se não for o ponto final)
        if rebal_date < end_date:
            buy_date = rebal_date # Aporte é feito no último dia útil do mês
            prices_on_buy = subset_prices.loc[buy_date]
            
            # Estratégia: Aporte de 1000
            investment_strategy = monthly_contribution
            
            for ticker in valid_tickers:
                amount_to_buy = investment_strategy * weights[ticker]
                
                # Compra e adiciona ao capital total
                if prices_on_buy[ticker] > 0:
                    shares = amount_to_buy / prices_on_buy[ticker]
                    capital_strategy.loc[buy_date] += shares * prices_on_buy[ticker]
                else:
                     shares = 0
                
                transactions_history.append({
                    'Mês/Data Aporte': buy_date.strftime('%Y-%m-%d'),
                    'Ticker': ticker,
                    'Aporte (R$)': f"{amount_to_buy:.2f}",
                    'Preço': f"{prices_on_buy[ticker]:.2f}",
                    'Ações Compradas': f"{shares:.2f}"
                })
            
            # Benchmark: Aporte de 1000 totalmente em BOVA11.SA
            shares_bench = monthly_contribution / prices_on_buy['BOVA11.SA']
            capital_benchmark.loc[buy_date] += shares_bench * prices_on_buy['BOVA11.SA']
            
            transactions_history.append({
                'Mês/Data Aporte': buy_date.strftime('%Y-%m-%d'),
                'Ticker': 'BOVA11.SA (Benchmark)',
                'Aporte (R$)': f"{monthly_contribution:.2f}",
                'Preço': f"{prices_on_buy['BOVA11.SA']:.2f}",
                'Ações Compradas': f"{shares_bench:.2f}"
            })
            
            last_idx = buy_date
        
        # C. Caso do primeiro ponto (i=0): Define o ponto inicial
        if i == 0:
            last_idx = rebal_date # Ponto de partida
            
    # 4. Combinar resultados
    curve = pd.DataFrame({
        'Strategy': capital_strategy[capital_strategy > 0].cumsum().ffill().dropna(),
        'BOVA11.SA': capital_benchmark[capital_benchmark > 0].cumsum().ffill().dropna()
    })
    
    # Ajustar a curva para usar a valorização contínua (substituir cumsum)
    # A lógica de valorização no loop já cuida da acumulação correta.
    # Vamos re-ajustar 'capital_strategy' e 'capital_benchmark' para uma série temporal única:
    
    # Criar uma série de valorização que preenche os dias entre os aportes/rebal.
    all_dates = subset_prices.index
    
    final_curve = pd.DataFrame(index=all_dates)
    final_curve['Strategy'] = 0.0
    final_curve['BOVA11.SA'] = 0.0

    current_capital_strat = 0.0
    current_capital_bench = 0.0
    
    all_dca_points = sorted(list(set([t['Mês/Data Aporte'] for t in transactions_history])))

    for k in range(len(all_dates)):
        day = all_dates[k]
        
        # 1. Aplicar aporte/rebalanceamento se for dia de compra
        if day.strftime('%Y-%m-%d') in all_dca_points:
            # Adiciona o valor aportado (aprox. 1000)
            current_capital_strat += monthly_contribution 
            current_capital_bench += monthly_contribution
            
        # 2. Aplicar retornos do dia anterior
        if k > 0:
            prev_day = all_dates[k-1]
            rets_day = subset_prices.loc[day].pct_change()
            
            # Retorno Estratégia
            ret_strat = rets_day[valid_tickers].dot(weights[valid_tickers].fillna(0))
            current_capital_strat *= (1 + ret_strat)
            
            # Retorno Benchmark
            ret_bench = rets_day['BOVA11.SA']
            current_capital_bench *= (1 + ret_bench)

        final_curve.loc[day, 'Strategy'] = current_capital_strat
        final_curve.loc[day, 'BOVA11.SA'] = current_capital_bench

    return final_curve.replace([np.inf, -np.inf], np.nan).dropna(how='all').ffill().dropna(), pd.DataFrame(transactions_history)


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
    
    use_vol_target = st.sidebar.checkbox("Usar Ponderação por Risco Inverso?", True)
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
            # Precisamos de dados suficientes para 252 dias e o cálculo de 2 anos (para o fator Residual Momentum)
            start_date_main = end_date - timedelta(days=730)
            
            prices = fetch_price_data(tickers, start_date_main, end_date)
            fundamentals = fetch_fundamentals(tickers)
            
            if prices.empty or fundamentals.empty:
                st.error("Não foi possível obter dados suficientes.")
                status.update(label="Erro!", state="error")
                return

            # 2. Cálculos e Ranking (Usa os preços completos)
            st.write("🧮 Calculando fatores e alocação estática...")
            res_mom = compute_residual_momentum(prices)
            fund_mom = compute_fundamental_momentum(fundamentals)
            val_score = compute_value_score(fundamentals)
            qual_score = compute_quality_score(fundamentals)

            df_master = pd.DataFrame(index=tickers)
            df_master['Res_Mom'] = res_mom
            df_master['Fund_Mom'] = fund_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            
            if 'sector' in fundamentals.columns: df_master['Sector'] = fundamentals['sector']
            df_master.dropna(thresh=2, inplace=True)

            cols_to_norm = ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']
            norm_cols = []
            for c in cols_to_norm:
                if c in df_master.columns:
                    new_col = f"{c}_Z"
                    df_master[new_col] = robust_zscore(df_master[c])
                    norm_cols.append(new_col)
            
            weights_dict = {
                'Res_Mom_Z': w_rm, 'Fund_Mom_Z': w_fm, 
                'Value_Z': w_val, 'Quality_Z': w_qual
            }
            
            final_df = build_composite_score(df_master, weights_dict)
            weights = construct_portfolio(final_df, prices, top_n, target_vol)

            status.update(label="Concluído!", state="complete", expanded=False)

        # --- OUTPUTS ---
        
        tab1, tab2, tab_dca, tab3 = st.tabs(["🏆 Ranking & Seleção", "📈 Backtest (Aporte Único)", "💰 Aportes Mensais DCA", "🔍 Detalhes dos Fatores"])
        
        with tab1:
            # ... (código da tab 1) ...
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
                    
                    st.metric("Soma da Alocação", f"{total_sum:.2%}")
                    
                    w_df["Peso"] = w_df["Peso"].map("{:.2%}".format)
                    st.table(w_df)
                    
                    fig_pie = px.pie(values=weights.values, names=weights.index, title="Distribuição")
                    st.plotly_chart(fig_pie, use_container_width=True)


        with tab2:
            # ... (código da tab 2) ...
            st.subheader("Performance Recente (Simulação de 1 Ano - Aporte Único)")
            st.info("Simula o desempenho de um aporte inicial na Estratégia vs. Benchmark.")
            
            if not weights.empty:
                curve = run_backtest(weights, prices)
                
                if not curve.empty and len(curve) > 1:
                    
                    daily_rets = curve.pct_change().dropna()
                    
                    # Estratégia
                    tot_ret_strat = curve['Strategy'].iloc[-1] - 1
                    vol_strat = daily_rets['Strategy'].std() * (252**0.5)
                    sharpe_strat = tot_ret_strat / vol_strat if vol_strat > 0 else 0

                    # Benchmark
                    tot_ret_bench = curve['BOVA11.SA'].iloc[-1] - 1
                    vol_bench = daily_rets['BOVA11.SA'].std() * (252**0.5)
                    sharpe_bench = tot_ret_bench / vol_bench if vol_bench > 0 else 0
                    
                    
                    st.markdown("### 🏆 Comparação de Métricas")
                    col_met1, col_met2, col_met3 = st.columns(3)
                    
                    col_met1.metric("Retorno Total (Estratégia)", f"{tot_ret_strat:.2%}", delta=f"vs. {tot_ret_bench:.2%} (Benchmark)")
                    col_met2.metric("Volatilidade Anual", f"{vol_strat:.2%}", delta=f"vs. {vol_bench:.2%} (Benchmark)", delta_color="inverse")
                    col_met3.metric("Sharpe Ratio (Anual)", f"{sharpe_strat:.2f}", delta=f"vs. {sharpe_bench:.2f} (Benchmark)")
                    
                    st.markdown("---")
                    
                    fig = px.line(curve, title="Curva de Equity: Estratégia vs BOVA11.SA")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Dados insuficientes para calcular o backtest no período.")
            else:
                st.warning("Nenhum ativo selecionado.")


        with tab_dca:
            st.subheader("💰 Simulação de Aportes Mensais (R$ 1000/mês)")
            st.info(f"Simulação nos **últimos 252 dias úteis**. Aporte ocorre no **último dia útil do mês** usando a alocação de hoje.")
            
            if not weights.empty:
                dca_curve, transactions = run_dca_backtest(weights, prices, lookback_days=252)
                
                if not dca_curve.empty:
                    
                    # O número de aportes é o número de meses no período (normalmente 12)
                    num_aportes = len(transactions['Mês/Data Aporte'].unique()) - 1 # Subtrai o benchmark
                    total_aportado = num_aportes * 1000
                    
                    # Valor final da Estratégia
                    final_value_strat = dca_curve['Strategy'].iloc[-1]
                    total_ret_strat = final_value_strat - total_aportado
                    
                    # Valor final do Benchmark
                    final_value_bench = dca_curve['BOVA11.SA'].iloc[-1]
                    total_ret_bench = final_value_bench - total_aportado
                    
                    
                    col_dca1, col_dca2, col_dca3 = st.columns(3)
                    col_dca1.metric("Total Aportado", f"R$ {total_aportado:,.2f}")
                    col_dca2.metric("Valor Final (Estratégia)", f"R$ {final_value_strat:,.2f}", delta=f"R$ {total_ret_strat:,.2f} de lucro")
                    col_dca3.metric("Valor Final (Benchmark)", f"R$ {final_value_bench:,.2f}", delta=f"R$ {total_ret_bench:,.2f} de lucro")
                    
                    
                    fig_dca = px.line(dca_curve, title="Curva de Aportes Mensais (DCA): Estratégia vs BOVA11.SA")
                    st.plotly_chart(fig_dca, use_container_width=True)
                    

                    st.markdown("### 🛒 Histórico de Compras Mensais na Estratégia")
                    st.dataframe(transactions.head(num_aportes * top_n)) # Limita a exibição para clareza
                else:
                    st.warning("Dados insuficientes para realizar a simulação DCA.")


        with tab3:
            # ... (código da tab 3) ...
            st.subheader("Correlação entre Fatores (Normalizados)")
            if norm_cols:
                corr = final_df[norm_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Mapa de Calor de Correlação")
                st.plotly_chart(fig_corr)
            
            st.subheader("Dados Fundamentais Brutos")
            st.dataframe(fundamentals)

if __name__ == "__main__":
    main()
