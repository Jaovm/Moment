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
    page_title="Quant Factor Lab: Backtest de Aporte",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Universo Amplo para Normalização (Comparação Externa)
# ATENÇÃO: Esta lista é usada para normalizar os fatores (z-score)
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
    clean_tickers = [t for t in all_tickers if t != 'BOVA11.SA']
    
    data = []
    progress_bar = st.progress(0)
    total = len(clean_tickers)
    
    for i, t in enumerate(clean_tickers):
        try:
            ticker_obj = yf.Ticker(t)
            info = ticker_obj.info
            
            data.append({
                'ticker': t,
                'Sector': info.get('sector', info.get('Sector', 'Unknown')), 
                'forwardPE': info.get('forwardPE', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan)
            })
        except Exception:
            pass
            
        if (i + 1) % 5 == 0: 
            progress_bar.progress(min((i + 1) / total, 1.0))
        
    progress_bar.empty()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data).set_index('ticker')
    cols_to_check = [c for c in df.columns if c not in ['Sector']]
    df = df.dropna(subset=cols_to_check, how='all')
    
    return df

# ==============================================================================
# MÓDULO 2: CÁLCULO DE FATORES (Math & Logic)
# ==============================================================================

def compute_residual_momentum(price_df: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Calcula Residual Momentum (Alpha) vs BOVA11.SA."""
    if price_df.empty: return pd.Series(dtype=float)
    
    df = price_df.copy()
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
    """Z-Score Robusto (Mediana e MAD)."""
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty: return pd.Series(np.nan, index=series.index)
        
    median = series.median()
    mad = (series - median).abs().median()
    
    z = pd.Series(np.nan, index=series.index)
    
    if mad == 0: 
        z.loc[series.index] = 0
    else:
        z.loc[series.index] = (series - median) / (mad * 1.4826)
        
    return z.clip(-3, 3)

def build_composite_score(df_master: pd.DataFrame, weights: dict) -> pd.DataFrame:
    """Calcula score final ponderado."""
    df = df_master.copy()
    df['Composite_Score'] = 0.0
    for factor_col, weight in weights.items():
        if factor_col in df.columns:
            # fillna(0) para tratar os NaNs restantes (tickers com alguns fatores ausentes)
            df['Composite_Score'] += df[factor_col].fillna(0) * weight
            
    return df.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 4: PORTFOLIO & BACKTEST (FIXED WEIGHTS & REBALANCING)
# ==============================================================================

def construct_portfolio(ranked_df: pd.DataFrame, prices: pd.DataFrame, top_n: int, vol_target: float = None):
    """Define pesos do portfólio (usado para in-sample e para rebalanceamento)."""
    available_tickers = [t for t in ranked_df.index if t in prices.columns]
    selected = ranked_df.loc[available_tickers].head(top_n).index.tolist()
    
    if not selected: return pd.Series()

    if vol_target is not None:
        recent_rets = prices[selected].pct_change().tail(63) 
        if recent_rets.empty: return pd.Series(1.0/len(selected), index=selected)
        
        vols = recent_rets.std()
        vols = vols.replace(0, 1e-6) 
        
        raw_weights_inv = 1 / vols
        weights = raw_weights_inv / raw_weights_inv.sum() 
    else:
        weights = pd.Series(1.0/len(selected), index=selected)
        
    return weights


def run_rebalancing_backtest(
    prices: pd.DataFrame, 
    fundamentals_broad: pd.DataFrame, 
    weights_dict: dict, 
    top_n: int, 
    monthly_contribution: float = 1000.0,
    rebalance_freq: str = 'M'
):
    """
    Simula um backtest de rebalanceamento mensal com aporte constante.
    O ranking é recalculado a cada rebalanceamento.
    """
    
    rebalance_dates = prices.index.to_series().resample(rebalance_freq).last().dropna()
    
    if len(rebalance_dates) < 2:
        return pd.DataFrame(), pd.DataFrame(), "Dados insuficientes para backtest de rebalanceamento."

    first_price_date = prices.index[0]
    
    rebalance_dates = rebalance_dates[rebalance_dates > first_price_date]
    rebalance_dates = pd.Index([first_price_date] + rebalance_dates.tolist())
    
    # Inicialização
    portfolio = {'Cash': 0.0}
    holdings = pd.Series(0.0, index=prices.columns).fillna(0) 
    equity_curve = []
    trade_log = []
    
    # 2. Loop de Rebalanceamento
    for i in range(len(rebalance_dates)):
        
        current_date = rebalance_dates[i]
        end_period = rebalance_dates[i+1] if i < len(rebalance_dates) - 1 else prices.index[-1]
        
        # --- Aporte ---
        portfolio['Cash'] += monthly_contribution
        
        # --- Recálculo de Fatores e Ranking ---
        
        prices_upto_current = prices[prices.index <= current_date]
        
        # Recalcula Momentum Residual (dinâmico)
        res_mom = compute_residual_momentum(prices_upto_current)
        
        # Fundamentos (estático)
        fund_mom = compute_fundamental_momentum(fundamentals_broad)
        val_score = compute_value_score(fundamentals_broad)
        qual_score = compute_quality_score(fundamentals_broad)

        df_master = pd.DataFrame(index=fundamentals_broad.index)
        df_master['Res_Mom'] = res_mom
        df_master['Fund_Mom'] = fund_mom
        df_master['Value'] = val_score
        df_master['Quality'] = qual_score
        df_master['Sector'] = fundamentals_broad['Sector']

        # Normalização (Global e Setorial)
        df_master_norm = df_master.copy()
        
        z_cols = []
        for col in ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']:
            if col in df_master_norm.columns and 'Sector' in df_master_norm.columns:
                new_col = f'{col}_Z'
                
                if col in ['Value', 'Quality']:
                    df_master_norm[new_col] = df_master_norm.groupby('Sector')[col].transform(robust_zscore)
                else:
                    df_master_norm[new_col] = robust_zscore(df_master_norm[col])
                
                if new_col in df_master_norm.columns:
                    z_cols.append(new_col)

        # --------------------- FIX DO KEYERROR ---------------------
        # Filtra tickers sem NENHUM score calculado (todos os Z-scores são NaN)
        if z_cols:
            df_to_score = df_master_norm.dropna(subset=z_cols, how='all')
        else:
            df_to_score = pd.DataFrame(columns=df_master_norm.columns) # DataFrame vazio se não houver scores
        
        # O build_composite_score calcula o Composite_Score
        final_df = build_composite_score(df_to_score, weights_dict) 
        # ------------------- FIM DO FIX ---------------------------

        # Seleção e Pesos
        weights_target = construct_portfolio(final_df, prices_upto_current, top_n, vol_target=0.15)
        
        # Filtra apenas os ativos disponíveis
        valid_tickers_for_trade = [t for t in weights_target.index if t in prices_upto_current.columns]
        weights_target = weights_target.loc[valid_tickers_for_trade]

        # --- Execução do Rebalanceamento ---
        
        current_price = prices_upto_current.loc[current_date]
        
        # 1. Valor da Carteira antes do rebalance
        port_value_before_rebal = portfolio['Cash'] + (holdings.reindex(current_price.index, fill_value=0).fillna(0) * current_price).sum()
        total_value = port_value_before_rebal 
        
        # 2. Pesos alvo em valor monetário
        target_value = total_value * weights_target.reindex(current_price.index, fill_value=0).fillna(0)
        
        # 3. Posição atual em valor monetário
        current_value = holdings.reindex(current_price.index, fill_value=0).fillna(0) * current_price
        
        # 4. Cálculo da Ação (Compra/Venda) - Rebalanceamento
        trades_value = target_value - current_value
        
        new_cash = portfolio['Cash']
        new_holdings = holdings.reindex(current_price.index, fill_value=0).fillna(0).copy()
        transaction_records = []
        
        # Vendas (negative trades_value)
        for t in trades_value.index:
            if trades_value.loc[t] < 0 and new_holdings.loc[t] > 0 and current_price.loc[t] > 0:
                shares_to_sell_target = np.floor(-trades_value.loc[t] / current_price.loc[t])
                shares_to_sell = min(new_holdings.loc[t], shares_to_sell_target)
                
                if shares_to_sell > 0:
                    amount = shares_to_sell * current_price.loc[t]
                    new_holdings.loc[t] -= shares_to_sell
                    new_cash += amount
                    
                    transaction_records.append({
                        'Date': current_date, 'Ticker': t, 'Action': 'SELL',
                        'Shares': shares_to_sell, 'Price': current_price.loc[t], 'Amount': amount
                    })

        # Compras (positive trades_value)
        new_total_value = new_cash + (new_holdings * current_price).sum()
        target_value = new_total_value * weights_target.reindex(current_price.index, fill_value=0).fillna(0)
        current_value = new_holdings * current_price
        trades_value_new = target_value - current_value

        for t in trades_value_new.index:
            if trades_value_new.loc[t] > 0 and current_price.loc[t] > 0 and new_cash > 0:
                buy_limit_value = min(trades_value_new.loc[t], new_cash)
                shares_to_buy = np.floor(buy_limit_value / current_price.loc[t])
                
                if shares_to_buy > 0:
                    amount = shares_to_buy * current_price.loc[t]
                    new_holdings.loc[t] += shares_to_buy
                    new_cash -= amount
                    
                    transaction_records.append({
                        'Date': current_date, 'Ticker': t, 'Action': 'BUY',
                        'Shares': shares_to_buy, 'Price': current_price.loc[t], 'Amount': amount
                    })

        portfolio['Cash'] = new_cash
        holdings = new_holdings
        trade_log.extend(transaction_records)
        
        # --- Cálculo da Equity Curve (do rebalanceamento atual até o próximo) ---
        
        period_prices = prices[(prices.index >= current_date) & (prices.index <= end_period)]
        
        for price_date in period_prices.index:
            market_value = (holdings.reindex(price_date.index, fill_value=0).fillna(0) * prices.loc[price_date, holdings.index]).sum()
            total_equity = market_value + portfolio['Cash']
            
            bova_price = prices.loc[price_date, 'BOVA11.SA'] if 'BOVA11.SA' in prices.columns else np.nan
            
            equity_curve.append({
                'Date': price_date,
                'Strategy': total_equity,
                'BOVA11.SA': bova_price
            })
            
    # --- Pós-processamento ---
    
    equity_df = pd.DataFrame(equity_curve).drop_duplicates(subset='Date').set_index('Date').sort_index()
    
    if not equity_df.empty and 'BOVA11.SA' in equity_df.columns:
        start_value = equity_df['Strategy'].iloc[0]
        
        start_bova_price = equity_df['BOVA11.SA'].iloc[0]
        equity_df['BOVA11.SA'] = (equity_df['BOVA11.SA'] / start_bova_price) * start_value
        
    trade_df = pd.DataFrame(trade_log)
    trade_df['Date'] = pd.to_datetime(trade_df['Date']).dt.date
    
    return equity_df, trade_df, ""


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
    
    run_btn = st.sidebar.button("🚀 Rodar Análise", type="primary")

    # --- MAIN LOGIC ---
    if run_btn:
        if not user_tickers:
            st.error("Por favor, insira pelo menos um ticker.")
            return

        with st.status("Executando Pipeline Quant...", expanded=True) as status:
            
            st.write("📥 Baixando dados de mercado e fundamentais (Comparação Ampla)...")
            end_date = datetime.now()
            start_date_backtest = end_date - timedelta(days=730) 
            
            prices = fetch_price_data(user_tickers, start_date_backtest, end_date) 
            fundamentals_broad = fetch_fundamentals(user_tickers)
            
            if prices.empty or fundamentals_broad.empty:
                st.error("Não foi possível obter dados suficientes. Verifique os tickers.")
                status.update(label="Erro!", state="error")
                return

            st.write("🧮 Calculando fatores multifatoriais (Snapshot)...")
            
            # Calculo do Snapshot Atual (Para Tab 1, 3, 4, 5)
            res_mom = compute_residual_momentum(prices)
            fund_mom = compute_fundamental_momentum(fundamentals_broad)
            val_score = compute_value_score(fundamentals_broad)
            qual_score = compute_quality_score(fundamentals_broad)
            
            df_master = pd.DataFrame(index=fundamentals_broad.index)
            df_master['Res_Mom'] = res_mom
            df_master['Fund_Mom'] = fund_mom
            df_master['Value'] = val_score
            df_master['Quality'] = qual_score
            if 'Sector' in fundamentals_broad.columns:
                df_master['Sector'] = fundamentals_broad['Sector']
            
            valid_user_tickers = [t for t in user_tickers if t in df_master.index]
            df_master_filtered = df_master.loc[valid_user_tickers].copy()
            
            if df_master_filtered.empty:
                st.error("Nenhum ticker válido encontrado nos dados fundamentais.")
                status.update(label="Erro!", state="error")
                return

            # Normalização (Global e Setorial)
            norm_cols = []
            for col in ['Res_Mom', 'Fund_Mom', 'Value', 'Quality']:
                new_col = f'{col}_Z'
                
                if col in ['Value', 'Quality'] and 'Sector' in df_master.columns:
                    df_master[new_col] = df_master.groupby('Sector')[col].transform(robust_zscore)
                else:
                    df_master[new_col] = robust_zscore(df_master[col])

                df_master_filtered[new_col] = df_master.loc[df_master_filtered.index, new_col]
                norm_cols.append(new_col)

            df_master_filtered.dropna(subset=[c for c in norm_cols if c in df_master_filtered.columns], how='all', inplace=True)

            weights_dict = {
                'Res_Mom_Z': w_rm, 'Fund_Mom_Z': w_fm, 
                'Value_Z': w_val, 'Quality_Z': w_qual
            }
            
            final_df = build_composite_score(df_master_filtered, weights_dict)
            weights = construct_portfolio(final_df, prices, top_n, 0.15 if use_vol_target else None)
            
            # Merge seguro para dados fundamentais + scores
            cols_to_merge = [c for c in final_df.columns if c not in fundamentals_broad.columns]
            fundamentals_final = pd.merge(
                fundamentals_broad.loc[final_df.index],
                final_df[cols_to_merge],
                left_index=True,
                right_index=True,
                how='inner'
            )
            
            st.write("🔄 Executando Backtest de Rebalanceamento Mensal (Aporte R$1000)...")
            equity_curve_rebal, trade_log, error_msg = run_rebalancing_backtest(
                prices, fundamentals_broad, weights_dict, top_n, 
                monthly_contribution=1000.0, rebalance_freq='M'
            )

            status.update(label="Concluído!", state="complete", expanded=False)

        # --- OUTPUTS ---
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏆 Ranking & Seleção", 
            "🔄 Backtest com Aporte Mensal", 
            "📈 Backtest (In-Sample)", 
            "🔍 Dados Fundamentais", 
            "📜 Justificativa da Seleção"
        ])
        
        with tab1:
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Top Picks (Snapshot Atual)")
                st.markdown(f"**Nota:** Comparação feita contra universo de {len(fundamentals_broad)} ativos.")
                show_cols = ['Composite_Score', 'Sector'] + norm_cols
                st.dataframe(
                    final_df[show_cols].head(top_n).style.background_gradient(cmap='RdYlGn', subset=['Composite_Score']),
                    height=400, width='stretch'
                )
            with col2:
                st.subheader("Alocação Sugerida")
                if not weights.empty:
                    w_df = weights.to_frame(name="Peso")
                    st.metric("Soma da Alocação", f"{weights.sum():.2%}")
                    w_df["Peso"] = w_df["Peso"].map("{:.2%}".format)
                    st.table(w_df)

        with tab2:
            st.subheader("Simulação de Aporte Mensal (R$1000)")
            
            if not equity_curve_rebal.empty:
                
                # Cálculo de Métricas (Aporte Contínuo)
                # Soma dos aportes
                total_contribution = 1000.0 * (len(equity_curve_rebal.index.to_series().resample('M').last().dropna()))
                
                final_value = equity_curve_rebal['Strategy'].iloc[-1]
                bova_final_value = equity_curve_rebal['BOVA11.SA'].iloc[-1]
                
                # Retorno sobre o capital investido
                ret_strategy = (final_value / total_contribution) - 1
                ret_bova = (bova_final_value / total_contribution) - 1

                m1, m2, m3 = st.columns(3)
                m1.metric("Aporte Total", f"R$ {total_contribution:,.2f}")
                m2.metric("Valor Final Estratégia", f"R$ {final_value:,.2f}")
                m3.metric("Valor Final BOVA11", f"R$ {bova_final_value:,.2f}")
                
                fig = px.line(equity_curve_rebal, title="Evolução da Carteira (Com Aporte)")
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("Histórico de Compras e Vendas (Trade Log)")
                st.dataframe(trade_log)

            else:
                st.warning(f"Dados insuficientes para backtest de rebalanceamento. {error_msg}")


        with tab3:
            st.subheader("Simulação (Backtest In-Sample - Pesos Fixos)")
            
            if not weights.empty:
                def run_backtest_simple(weights: pd.Series, prices: pd.DataFrame, lookback_days: int = 504):
                    valid_tickers = [t for t in weights.index if t in prices.columns]
                    if not valid_tickers: return pd.DataFrame()
                        
                    subset = prices.tail(lookback_days)
                    rets = subset.pct_change().dropna()
                    
                    port_ret = rets[valid_tickers].dot(weights[valid_tickers].fillna(0))
                    BVSP_ret = rets['BOVA11.SA'] if 'BOVA11.SA' in rets.columns else pd.Series(0, index=rets.index)
                    
                    daily_rets = pd.DataFrame({'Strategy': port_ret, 'BOVA11.SA': BVSP_ret})
                    cumulative = (1 + daily_rets).cumprod()
                    return cumulative.dropna()

                curve = run_backtest_simple(weights, prices)
                if not curve.empty and len(curve) > 1:
                    tot_ret = curve['Strategy'].iloc[-1] - 1
                    daily_rets = curve.pct_change().dropna()
                    vol = daily_rets['Strategy'].std() * (252**0.5)
                    sharpe = tot_ret / vol if vol > 0 else 0
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Retorno Total", f"{tot_ret:.2%}")
                    m2.metric("Volatilidade", f"{vol:.2%}")
                    m3.metric("Sharpe", f"{sharpe:.2f}")
                    
                    fig = px.line(curve, title="Equity Curve (In-Sample)")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Dados insuficientes para backtest in-sample.")
            else:
                st.warning("Nenhum ativo selecionado.")


        with tab4:
            st.subheader("Dados Fundamentais Brutos (Com Score)")
            st.dataframe(fundamentals_final.loc[final_df.index])
            
            if norm_cols:
                st.subheader("Correlação (Fatores Normalizados)")
                corr = final_df[[c for c in norm_cols if c in final_df.columns]].corr()
                fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr)

        with tab5:
            st.subheader("Justificativa Detalhada (Top Picks)")
            
            detailed_df = fundamentals_final.sort_values('Composite_Score', ascending=False).head(top_n)
            
            for ticker in detailed_df.index:
                row = detailed_df.loc[ticker]
                st.markdown(f"### 📈 {ticker} - Score: {row.get('Composite_Score', 0):.2f}")
                st.markdown(f"**Setor:** {row.get('Sector', 'N/A')}")
                
                justification = []
                
                def analyze_factor(factor_name, label_name, high_desc, low_desc):
                    val = row.get(factor_name, 0)
                    if pd.isna(val) or f'{factor_name}_Z' not in row: 
                        justification.append(f"- **{label_name}:** Dado fundamental indisponível.")
                        return

                    z_val = row.get(f'{factor_name}_Z', 0)

                    if z_val > 0.5: justification.append(f"- **{label_name} ({z_val:.2f}):** {high_desc}")
                    elif z_val > 0.0: justification.append(f"- **{label_name} ({z_val:.2f}):** Acima da média.")
                    else: justification.append(f"- **{label_name} ({z_val:.2f}):** {low_desc}")

                analyze_factor('Res_Mom', 'Residual Momentum', 'Forte Alpha (retorno acima do Ibov).', 'Preço sem força relativa recente.')
                analyze_factor('Fund_Mom', 'Fundamental Momentum', 'Lucros/Receita acelerando forte (topo do mercado).', 'Crescimento estagnado/negativo vs mercado.')
                analyze_factor('Value', 'Valor (vs Setor)', 'Muito descontado (P/L e P/VP baixos) vs pares.', 'Preço justo ou caro vs pares setoriais.')
                analyze_factor('Quality', 'Qualidade (vs Setor)', 'Alta eficiência (ROE) e solidez vs pares.', 'Qualidade na média ou abaixo dos pares.')

                st.info("\n".join(justification) if justification else "Sem dados suficientes para justificativa.")

if __name__ == "__main__":
    main()
