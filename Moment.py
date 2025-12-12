import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore') # Ignora avisos de pacotes (como statsmodels) para manter a interface limpa

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="Quant Factor Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Universo Amplo para Normalização (Comparação Externa)
# Mantido o universo da versão anterior para consistência com o pedido
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
            auto_adjust=True # Usar auto_adjust=True para backtest de longo prazo
        )['Close']
        
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
    # Usando st.progress para dar feedback
    progress_bar = st.progress(0, text="Baixando Fundamentos do Mercado...")
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
                'enterpriseToEbitda': info.get('enterpriseToEbitda', np.nan),
                'returnOnEquity': info.get('returnOnEquity', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
                'debtToEquity': info.get('debtToEquity', np.nan),
                'earningsGrowth': info.get('earningsGrowth', np.nan),
                'revenueGrowth': info.get('revenueGrowth', np.nan)
            })
        except Exception:
            pass 
            
        if (i + 1) % 5 == 0: 
            progress_bar.progress(min((i + 1) / total, 1.0), text=f"Baixando {t}...")
        
    progress_bar.empty()
    
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data).set_index('ticker')
    cols_to_check = [c for c in df.columns if c != 'Sector']
    df = df.dropna(subset=cols_to_check, how='all')
    
    return df

# ==============================================================================
# MÓDULO 2 & 3: CÁLCULO DE FATORES E SCORING
# ==============================================================================

# Função Z-Score Robusto
def robust_zscore(series: pd.Series) -> pd.Series:
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

# Função para calcular Residual Momentum em um ponto no tempo
def compute_residual_momentum_at_date(price_df_snapshot: pd.DataFrame, lookback=12, skip=1) -> pd.Series:
    """Calcula Residual Momentum (Alpha) no último ponto do snapshot."""
    if price_df_snapshot.empty or 'BOVA11.SA' not in price_df_snapshot.columns: 
        return pd.Series(dtype=float)
    
    df = price_df_snapshot.copy()
    
    # Resample mensal para o cálculo
    try:
        monthly = df.resample('ME').last()
    except:
        monthly = df.resample('M').last()

    rets = monthly.pct_change().dropna()
    
    market = rets['BOVA11.SA']
    scores = {}
    window = lookback + skip
    
    # Garantir que temos dados suficientes
    if len(rets) < window: return pd.Series(dtype=float)
    
    for ticker in rets.columns:
        if ticker == 'BOVA11.SA': continue
        
        y = rets[ticker].tail(window)
        x = market.tail(window)
            
        try:
            # Statsmodels exige que y e x sejam do mesmo tamanho
            X = sm.add_constant(x.values)
            model = sm.OLS(y.values, X).fit()
            resid = model.resid[:-skip]
            
            # Média dos resíduos normalizada pela volatilidade
            scores[ticker] = np.sum(resid) / np.std(resid) if np.std(resid) > 0 else 0
        except Exception:
            scores[ticker] = 0
            
    return pd.Series(scores, name='Res_Mom')

# Função unificada para gerar o ranking (adaptada para o backtest)
def get_ranking(prices_snapshot: pd.DataFrame, fundamentals_broad: pd.DataFrame, user_tickers: list, weights_dict: dict) -> pd.DataFrame:
    """Calcula todos os fatores e ranking em um ponto no tempo."""
    
    # 1. Fatores Momentum (Calculados no Ponto no Tempo)
    res_mom = compute_residual_momentum_at_date(prices_snapshot)
    
    # 2. Fatores Fundamentais (Usam o snapshot atual como proxy, limitação do yfinance)
    fund_mom = compute_fundamental_momentum(fundamentals_broad).rename('Fund_Mom')
    val_score = compute_value_score(fundamentals_broad).rename('Value')
    qual_score = compute_quality_score(fundamentals_broad).rename('Quality')

    # 3. Consolidação e Normalização
    df_master = pd.DataFrame(index=fundamentals_broad.index)
    df_master['Res_Mom'] = res_mom
    df_master['Fund_Mom'] = fund_mom
    df_master['Value'] = val_score
    df_master['Quality'] = qual_score
    df_master['Sector'] = fundamentals_broad['Sector']

    # Filtra tickers do usuário (ativos que podemos comprar)
    df_master_filtered = df_master.loc[df_master.index.intersection(user_tickers)].copy()
    
    if df_master_filtered.empty: return pd.DataFrame()

    cols_to_norm_sectorial = ['Value', 'Quality']
    cols_to_norm_global = ['Res_Mom', 'Fund_Mom'] 
    
    # Normalização é feita sobre o UNIVERSO AMPLO (df_master) e transferida para o filtrado
    for c in cols_to_norm_sectorial:
        if c in df_master.columns:
            new_col = f"{c}_Z"
            df_master[new_col] = df_master.groupby('Sector')[c].transform(robust_zscore)
            df_master_filtered[new_col] = df_master.loc[df_master_filtered.index, new_col]

    for c in cols_to_norm_global:
        if c in df_master.columns:
            new_col = f"{c}_Z"
            df_master[new_col] = robust_zscore(df_master[c])
            df_master_filtered[new_col] = df_master.loc[df_master_filtered.index, new_col]
            
    # Remove ativos sem scores
    final_cols = [k for k in weights_dict.keys() if k in df_master_filtered.columns]
    df_master_filtered.dropna(subset=final_cols, how='all', inplace=True)

    # 4. Score Composto
    df_master_filtered['Composite_Score'] = 0.0
    for factor_col, weight in weights_dict.items():
        if factor_col in df_master_filtered.columns:
            df_master_filtered['Composite_Score'] += df_master_filtered[factor_col].fillna(0) * weight
            
    return df_master_filtered.sort_values('Composite_Score', ascending=False)

# ==============================================================================
# MÓDULO 5: BACKTEST DCA (NOVO)
# ==============================================================================

def run_dca_backtest(prices: pd.DataFrame, fundamentals_broad: pd.DataFrame, user_tickers: list, 
                     weights_dict: dict, top_n: int, monthly_contribution: float) -> tuple:
    """
    Simula um backtest de aportes mensais com seleção de ativos por score.
    A rebalancagem da carteira só ocorre no aporte novo.
    """
    
    # 1. Preparação
    
    # Filtra apenas os tickers que têm preços
    valid_tickers = [t for t in user_tickers if t in prices.columns]
    if not valid_tickers: return pd.DataFrame(), pd.DataFrame(), pd.Series()

    # Identifica as datas de rebalanceamento (Último dia útil do mês)
    monthly_prices = prices.resample('ME').last().dropna(how='all')
    rebalance_dates = monthly_prices.index
    
    # Garante que temos pelo menos 13 meses de histórico para o primeiro cálculo de Momentum
    if len(rebalance_dates) < 13:
        st.warning("Histórico de preços insuficiente para backtest DCA (necessário > 12 meses).")
        return pd.DataFrame(), pd.DataFrame(), pd.Series()
    
    # Inicializa o controle do portfólio
    portfolio_value = pd.Series(0.0, index=prices.index)
    shares_held = pd.Series(0.0, index=valid_tickers)
    
    transaction_records = []
    
    # 2. Loop Mensal
    for i, date in enumerate(rebalance_dates):
        if date not in prices.index: continue # Pula se não houver preço neste dia
            
        # Calcula o valor do portfólio no início do mês (último dia útil anterior)
        if i > 0:
            previous_date = rebalance_dates[i-1]
            # Usa o preço do dia de rebalance anterior para avaliar o portfólio
            current_prices = prices.loc[prices.index <= previous_date].iloc[-1]
            portfolio_value.loc[date] = (shares_held * current_prices.reindex(shares_held.index)).sum()
        else:
            portfolio_value.loc[date] = 0.0 # Começa em 0

        # --- Aporte e Seleção (Rebalanceamento) ---
        cash_for_purchase = monthly_contribution
        
        # Snapshot de preços até a data atual (para Momentum Residual)
        prices_snapshot = prices.loc[prices.index <= date]

        # Gera o ranking para este ponto no tempo
        ranked_df = get_ranking(prices_snapshot, fundamentals_broad, valid_tickers, weights_dict)
        
        # Filtra os Top N selecionados
        selected_assets = ranked_df.head(top_n).index.tolist()
        
        # Preços de compra (Preço de fechamento no dia do aporte)
        buy_prices = prices.loc[date, selected_assets]
        
        if not selected_assets or buy_prices.empty:
            transaction_records.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Aporte (R$)': monthly_contribution,
                'Ações Compradas': "Nenhuma (Sem ativos válidos/preço)",
                'Alocação Total (R$)': 0.0,
                'Tickers': 'N/A'
            })
            # Se não comprou, o dinheiro fica de fora ou é "perdido" (simplificação)
            continue
