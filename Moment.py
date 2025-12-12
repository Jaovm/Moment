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
        # Puxa o histórico de preços ajustados
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

# Não atualizamos esta função pois o backtest mensal usará apenas preços
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
    # Pega apenas os últimos preços do mês para cálculo de retorno
    monthly = df.resample('ME').last()
    rets = monthly.pct_change().dropna()
    
    if 'BOVA11.SA' not in rets.columns: return pd.Series(dtype=float)
        
    market = rets['BOVA11.SA']
    scores = {}
    window = lookback + skip # Ex: 12 meses + 1 mês de skip
    
    for ticker in rets.columns:
        if ticker == 'BOVA11.SA': continue
        
        # Pega a janela de retornos e o benchmark correspondente
        y = rets[ticker].tail(window)
        x = market.tail(window)
        
        if len(y) < window: continue
            
        try:
            # Regressão OLS: Retorno do Ativo = Alpha + Beta * Retorno do Mercado + Epsilon
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            # Resíduos são os 'alphas' mensais
            resid = model.resid[:-skip] # Exclui o último mês para evitar look-ahead bias
            
            # Residual Momentum é a soma normalizada dos resíduos
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
    # Z-Score Robusto usa MAD (Median Absolute Deviation)
    z = (series - median) / (mad * 1.4826)
    return z.clip(-3, 3)

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Calcula score final ponderado."""
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    # Soma ponderada dos Z-Scores
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
        
        # 1. Calcular volatilidade histórica (3 meses / 63 dias)
        recent_rets = prices[selected].pct_change().tail(63)
        # Volatilidade Anualizada
        vols = recent_rets.std() * (252**0.5)
        vols[vols == 0] = 1e-6 # Evita divisão por zero
        
        # 2. Calcular Pesos de Risco Inverso (Proporcional ao inverso da Vol)
        raw_weights_inv = 1 / vols
        
        # 3. FORÇA A NORMALIZAÇÃO para 100%
        weights = raw_weights_inv / raw_weights_inv.sum() 
            
    else:
        # Pesos Iguais (Equal Weight)
        weights = pd.Series(1.0/len(selected), index=selected)
        
    return weights

def run_backtest(weights: pd.Series, prices: pd.DataFrame, lookback_days: int = 252):
    """
    Simula o desempenho do portfólio selecionado e do Benchmark no período dado (Backtest Simples).
    Retorna a Curva de Equity.
    """
    
    # 1. Preparação dos Dados
    subset = prices.tail(lookback_days)
    rets = subset.pct_change().dropna()
    
    # 2. Retorno do Benchmark
    if 'BOVA11.SA' in rets.columns:
        BVSP_ret = rets['BOVA11.SA']
    else:
        BVSP_ret = pd.Series(0, index=rets.index)
    
    # 3. Retorno do Portfólio 
    valid_tickers = [t for t in weights.index if t in prices.columns]
    
    if valid_tickers:
        # Produto escalar: Retorno Diário * Peso
        port_ret = rets[valid_tickers].dot(weights[valid_tickers].fillna(0))
    else:
        port_ret = pd.Series(0, index=rets.index)
        
    # 4. Cria DataFrame de retornos diários
    daily_rets = pd.DataFrame({'Strategy': port_ret, 'BOVA11.SA': BVSP_ret})
    
    # 5. Retorno Cumulativo (Curva de Equity)
    cumulative = (1 + daily_rets).cumprod()
    return cumulative.dropna()


def run_dca_backtest(
    all_prices: pd.DataFrame, 
    all_fundamentals: pd.DataFrame, 
    factor_weights: dict, 
    top_n: int, 
    dca_amount: float, 
    use_vol_target: bool,
    start_date: datetime,
    end_date: datetime
):
    """
    Simula um Backtest com Aportes Mensais (DCA) e rebalanceamento da Estratégia.
    """
    
    # 1. Configuração do Backtest
    dca_start = start_date + timedelta(days=30) # Começa 1 mês depois do start_date para garantir dados de lookback
    dates = all_prices.loc[dca_start:end_date].resample('MS').first().index.tolist()
    
    if not dates or len(dates) < 2:
        return pd.DataFrame(), pd.DataFrame()

    portfolio_value = pd.Series(0.0, index=all_prices.index)
    benchmark_value = pd.Series(0.0, index=all_prices.index)
    portfolio_holdings = {} # Ticker: [data, quantidade]
    benchmark_holdings = {'BOVA11.SA': 0.0} # Apenas quantidade de BOVA11.SA
    
    monthly_transactions = []
    
    # 2. Loop Mensal (Rebalanceamento e Aporte)
    for i, month_start in enumerate(dates):
        
        # --- Passo A: Avaliação (Usa dados *antes* do dia de rebalanceamento) ---
        
        # Data de corte para garantir que só usamos dados disponíveis até o dia anterior
        eval_date = month_start - timedelta(days=1)
        
        # Preços para cálculo do Momentum (13 meses antes do rebalanceamento)
        # O Residual Momentum precisa de 12 meses + 1 de skip, então olhamos 13 meses antes.
        mom_start = month_start - timedelta(days=395) 
        prices_for_mom = all_prices.loc[mom_start:eval_date] 
        
        # Preços para cálculo do Risco Inverso (63 dias antes do rebalanceamento)
        risk_start = month_start - timedelta(days=90)
        prices_for_risk = all_prices.loc[risk_start:eval_date]
        
        
        # 1. Recalcula Fatores e Pesos
        if not prices_for_mom.empty:
            res_mom = compute_residual_momentum(prices_for_mom)
        else:
            res_mom = pd.Series(dtype=float)
            
        # O fundamental momentum é estático na simulação (simplificação com dados yfinance)
        fund_mom = compute_fundamental_momentum(all_fundamentals)
        val_score = compute_value_score(all_fundamentals)
        qual_score = compute_quality_score(all_fundamentals)

        df_master = pd.DataFrame(index=all_prices.columns.drop('BOVA11.SA', errors='ignore'))
        df_master['Res_Mom'] = res_mom
        df_master['Fund_Mom'] = fund_mom
        df_master['Value'] = val_score
        df_master['Quality'] = qual_score
        
        df_master.dropna(thresh=2, inplace=True) # Remove se tiver poucos fatores
        
        # Normalização
        norm_cols = [c for c in ['Res_Mom', 'Fund_Mom', 'Value', 'Quality'] if c in df_master.columns]
        for c in norm_cols:
             df_master[f"{c}_Z"] = robust_zscore(df_master[c])
        
        weights_dict = {f"{k}_Z": v for k, v in factor_weights.items() if f"{k}_Z" in df_master.columns}
        
        final_df = build_composite_score(df_master, weights_dict)
        
        # Define os pesos do portfólio (EW ou Risco Inverso)
        current_weights = construct_portfolio(
            final_df, 
            prices_for_risk, # Usa preços recentes para cálculo de volatilidade
            top_n, 
            0.15 if use_vol_target else None # target_vol é apenas um placeholder aqui
        )
        
        # --- Passo B: Aporte e Compras ---
        
        # Pega o preço de abertura do dia de rebalanceamento (primeiro preço do mês)
        # Isso simula a compra na primeira oportunidade
        rebal_price = all_prices.loc[month_start].to_frame().T
        
        # 1. Estratégia (Aporte + Rebalanceamento)
        
        # Aporte de R$1000
        cash_for_strategy = dca_amount 
        
        # Itera sobre os ativos selecionados para comprar
        current_holdings = {t: q for t, q in portfolio_holdings.items() if q > 0}
        
        # Novos e antigos ativos que terão um peso desejado > 0
        all_rebal_tickers = list(set(current_weights.index.tolist() + list(current_holdings.keys())))
        
        
        # 2. Benchmark (Aporte)
        
        # Preço BOVA11.SA no dia da compra
        bova_price = rebal_price['BOVA11.SA'].iloc[0]
        
        if not np.isnan(bova_price) and bova_price > 0:
            # Quantidade de BOVA11.SA comprada
            q_bova = dca_amount / bova_price
            benchmark_holdings['BOVA11.SA'] += q_bova
            monthly_transactions.append({
                'Date': month_start,
                'Ticker': 'BOVA11.SA',
                'Action': 'Buy (DCA)',
                'Quantity': q_bova,
                'Price': bova_price,
                'Value_R$': dca_amount
            })
            
        # 3. Compra dos Ativos da Estratégia
        
        buy_value = 0
        # Distribui o cash entre os ativos selecionados de acordo com os pesos
        for ticker, weight in current_weights.items():
            if ticker in rebal_price.columns and not rebal_price[ticker].isna().iloc[0]:
                
                price = rebal_price[ticker].iloc[0]
                
                if price > 0 and weight > 0:
                    amount = cash_for_strategy * weight
                    quantity = amount / price
                    
                    # Atualiza o portfólio
                    portfolio_holdings[ticker] = portfolio_holdings.get(ticker, 0.0) + quantity
                    buy_value += amount
                    
                    monthly_transactions.append({
                        'Date': month_start,
                        'Ticker': ticker,
                        'Action': 'Buy (DCA)',
                        'Quantity': quantity,
                        'Price': price,
                        'Value_R$': amount
                    })
        
        # --- Passo C: Avaliação (Até o próximo rebalanceamento) ---
        
        # Último dia do mês para o valor final antes do rebalanceamento
        next_month_start = dates[i+1] if i < len(dates) - 1 else end_date
        valuation_dates = all_prices.loc[month_start:next_month_start].index
        
        # Simulação dia-a-dia da valorização
        for current_date in valuation_dates:
            
            # Valorização da Estratégia
            current_port_value = 0.0
            for ticker, quantity in portfolio_holdings.items():
                if ticker in all_prices.columns and current_date in all_prices.index:
                    price = all_prices.loc[current_date, ticker]
                    current_port_value += price * quantity
            
            portfolio_value[current_date] = current_port_value
            
            # Valorização do Benchmark
            current_bench_value = 0.0
            price_bova = all_prices.loc[current_date, 'BOVA11.SA']
            current_bench_value = price_bova * benchmark_holdings['BOVA11.SA']
            
            benchmark_value[current_date] = current_bench_value
        
    # Limpa valores zero e datas iniciais antes do primeiro aporte
    portfolio_value = portfolio_value[portfolio_value > 0].ffill().dropna()
    benchmark_value = benchmark_value[benchmark_value > 0].ffill().dropna()
    
    
    # Cria DataFrame final de Equity Curve
    equity_curve = pd.DataFrame({
        'Strategy_DCA': portfolio_value, 
        'BOVA11.SA_DCA': benchmark_value
    })

    return equity_curve, pd.DataFrame(monthly_transactions)


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
    
    st.sidebar.header("4. Simulação Mensal (DCA)")
    dca_amount = st.sidebar.number_input("Aporte Mensal (R$)", 100, 5000, 1000)
    dca_years = st.sidebar.slider("Anos de Backtest DCA", 1, 5, 3)
    
    run_btn = st.sidebar.button("🚀 Rodar Análise", type="primary")

    # --- MAIN LOGIC ---
    if run_btn:
        if not tickers:
            st.error("Por favor, insira pelo menos um ticker.")
            return

        with st.status("Executando Pipeline Quant...", expanded=True) as status:
            
            # 1. Dados (Puxa 2 anos para backtest simples + anos para DCA)
            end_date = datetime.now()
            # Puxa dados o suficiente para o Residual Momentum em todo o período DCA
            start_date = end_date - timedelta(days=365 * (dca_years + 1)) 
            prices = fetch_price_data(tickers, start_date, end_date)
            # Fundamentos estáticos (usados para a última data disponível)
            fundamentals = fetch_fundamentals(tickers) 
            
            if prices.empty or fundamentals.empty:
                st.error("Não foi possível obter dados suficientes.")
                status.update(label="Erro!", state="error")
                return
            
            # Pesos para o cálculo
            weights_dict_dca = {
                'Res_Mom': w_rm, 'Fund_Mom': w_fm, 
                'Value': w_val, 'Quality': w_qual
            }

            # 2. Cálculos e Ranking (Para a data atual)
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
            
            weights_dict_live = {
                'Res_Mom_Z': w_rm, 'Fund_Mom_Z': w_fm, 
                'Value_Z': w_val, 'Quality_Z': w_qual
            }
            
            final_df = build_composite_score(df_master, weights_dict_live)
            weights = construct_portfolio(final_df, prices, top_n, target_vol)
            
            # 3. Executa Backtest DCA
            dca_curve, dca_transactions = run_dca_backtest(
                prices,
                fundamentals, # Uso de fundamentos estáticos para simplificação
                weights_dict_dca,
                top_n,
                dca_amount,
                use_vol_target,
                end_date - timedelta(days=365 * dca_years),
                end_date
            )

            status.update(label="Concluído!", state="complete", expanded=False)

        # --- OUTPUTS ---
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "🏆 Ranking & Seleção (Atual)", 
            "📈 Backtest (In-Sample)", 
            "💰 Backtest DCA (Aportes Mensais)",
            "🔍 Detalhes dos Fatores"
        ])
        
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
                    
                    st.metric("Soma da Alocação", f"{total_sum:.2%}")
                    
                    w_df["Peso"] = w_df["Peso"].map("{:.2%}".format)
                    st.table(w_df)
                    
                    fig_pie = px.pie(values=weights.values, names=weights.index, title="Distribuição")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tab2:
            st.subheader("Performance Recente (Simulação de 1 Ano)")
            
            if not weights.empty:
                # O Backtest Simples usa apenas o último conjunto de pesos
                curve = run_backtest(weights, prices, lookback_days=252)
                
                if not curve.empty and len(curve) > 1:
                    
                    # CÁLCULO DAS MÉTRICAS
                    daily_rets = curve.pct_change().dropna()
                    
                    tot_ret_strat = curve['Strategy'].iloc[-1] - 1
                    vol_strat = daily_rets['Strategy'].std() * (252**0.5)
                    sharpe_strat = tot_ret_strat / vol_strat if vol_strat > 0 else 0

                    tot_ret_bench = curve['BOVA11.SA'].iloc[-1] - 1
                    vol_bench = daily_rets['BOVA11.SA'].std() * (252**0.5)
                    sharpe_bench = tot_ret_bench / vol_bench if vol_bench > 0 else 0
                    
                    
                    # EXIBIÇÃO DAS MÉTRICAS
                    
                    st.markdown("### 🏆 Comparação de Métricas")
                    col_met1, col_met2, col_met3 = st.columns(3)
                    
                    col_met1.metric("Retorno Total (Estratégia)", f"{tot_ret_strat:.2%}", delta=f"vs. {tot_ret_bench:.2%} (Benchmark)")
                    col_met2.metric("Volatilidade Anual", f"{vol_strat:.2%}", delta=f"vs. {vol_bench:.2%} (Benchmark)", delta_color="inverse")
                    col_met3.metric("Sharpe Ratio (Anual)", f"{sharpe_strat:.2f}", delta=f"vs. {sharpe_bench:.2f} (Benchmark)")
                    
                    st.markdown("---")
                    
                    fig = px.line(curve, title="Equity Curve (Compra Única): Estratégia vs BOVA11.SA")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Dados insuficientes para calcular o backtest no período.")
            else:
                st.warning("Nenhum ativo selecionado.")

        with tab3:
            st.subheader(f"Performance com Aportes Mensais (R${dca_amount:,.2f} DCA)")
            
            if not dca_curve.empty and len(dca_curve) > 1:
                
                # CÁLCULO DAS MÉTRICAS DCA
                
                # Último valor e valor inicial (que deve ser 0 para DCA)
                final_strat_value = dca_curve['Strategy_DCA'].iloc[-1]
                final_bench_value = dca_curve['BOVA11.SA_DCA'].iloc[-1]
                
                total_months = len(dca_transactions['Date'].unique())
                total_invested = total_months * dca_amount
                
                total_return_strat = final_strat_value - total_invested
                total_return_bench = final_bench_value - total_invested
                
                # Exibe o valor final
                st.markdown("### 💰 Resultado Final")
                col_dca1, col_dca2, col_dca3 = st.columns(3)
                col_dca1.metric("Total Investido", f"R${total_invested:,.2f}")
                col_dca2.metric("Valor Final (Estratégia)", f"R${final_strat_value:,.2f}", delta=f"Ganho: R${total_return_strat:,.2f}")
                col_dca3.metric("Valor Final (Benchmark)", f"R${final_bench_value:,.2f}", delta=f"Ganho: R${total_return_bench:,.2f}")
                
                st.markdown("---")
                
                # Gráfico de Equity
                fig_dca = px.line(dca_curve, title="Equity Curve (DCA): Estratégia vs BOVA11.SA")
                fig_dca.update_layout(yaxis_title="Valor Total do Portfólio (R$)")
                st.plotly_chart(fig_dca, use_container_width=True)
                
                # Tabela de Aportes
                st.subheader("Aportes e Seleções Mensais")
                dca_transactions['Date'] = dca_transactions['Date'].dt.strftime('%Y-%m-%d')
                dca_transactions['Price'] = dca_transactions['Price'].map('R${:,.2f}'.format)
                dca_transactions['Value_R$'] = dca_transactions['Value_R$'].map('R${:,.2f}'.format)
                dca_transactions['Quantity'] = dca_transactions['Quantity'].map('{:,.4f}'.format)
                
                st.dataframe(dca_transactions.set_index('Date'), height=300)
                
            else:
                st.warning("Dados insuficientes para calcular o backtest DCA no período.")

        with tab4:
            st.subheader("Correlação entre Fatores (Normalizados)")
            if norm_cols:
                corr = final_df[norm_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Mapa de Calor de Correlação")
                st.plotly_chart(fig_corr)
            
            st.subheader("Dados Fundamentais Brutos (Estáticos)")
            st.dataframe(fundamentals)

if __name__ == "__main__":
    main()
