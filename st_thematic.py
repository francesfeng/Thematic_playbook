import altair as alt
import streamlit as st
import pandas as pd
import numpy as np
import SessionState
from pandas.tseries.offsets import DateOffset

from streamlit.server.server import Server
import streamlit.report_thread as ReportThread


from streamlit_vega_lite import vega_lite_component, altair_component
pd.options.display.float_format = '{:,.0f}'.format

st.set_page_config(
     page_title="Thematic Playbook",
     page_icon=":sparkles:",
     layout="wide",
     #initial_sidebar_state="auto",
 )
#---------------------------------------------------------------------------------------------------Import data
@st.cache
def upload():
    price = pd.read_csv('to_upload/etf_wk.csv', index_col=0)
    etfs = pd.read_csv('to_upload/etf_ticker.csv', index_col=0)
    holdings = pd.read_csv('to_upload/holdings.csv', index_col=0)
    underlyings = pd.read_csv('to_upload/underlyings.csv', index_col=0)
    ratios = pd.read_csv('to_upload/ratios_ts_q.csv', index_col=0)
    type = pd.read_csv('to_upload/ratio_type.csv', index_col=0)
    stock_price = pd.read_csv('to_upload/price_wk.csv', index_col=0)
    crisis = pd.read_csv('to_upload/list_crisis.csv', index_col = 0)
    return_periods = pd.read_csv('to_upload/return_periods.csv', index_col = 0)

    price['Date'] = pd.to_datetime(price['Date'])
    ratios['Date'] = pd.to_datetime(ratios['Date'])
    #underlyings['Date'] = pd.to_datetime(underlyings['Date'])
    stock_price['Date'] = pd.to_datetime(stock_price['Date'])

    crisis['Start Date'] = pd.to_datetime(crisis['Start Date'], format='%d/%m/%Y')
    crisis['End Date'] = pd.to_datetime(crisis['End Date'], format='%d/%m/%Y')

    listmap= {}
    listmap['Country'] = underlyings.groupby('Country')['Ticker'].count().sort_values(ascending=False).index
    listmap['Industry'] = underlyings.groupby('Industry')['Ticker'].count().sort_values(ascending=False).index
    listmap['Sector'] = underlyings.groupby('Sector')['Ticker'].count().sort_values(ascending=False).index
    listmap['Size'] = underlyings.groupby('Size')['Ticker'].count().sort_values(ascending=False).index
    listmap['Business Activity'] = underlyings.groupby('Business Activity')['Ticker'].count().sort_values(ascending=False).index.tolist()

    q_latest = 'Q4 2020'
    ratio_latest =ratios[ratios['Quarter'] == q_latest ]
    ratios_des = ratio_latest.groupby('Short Name')['Value'].agg([np.min, np.max])
    ratios_des.columns = ['min','max']
    ratios_des['bins'] = round((ratios_des['max'] - ratios_des['min'])/100,1)
    listmap['Ratio'] = ratios_des

    ratios_p = ratio_latest.pivot_table(index = 'Ticker', columns = 'Short Name', values ='Value')
    underlying_map = pd.merge(underlyings, ratios_p, how='left', on='Ticker')

    return (price, etfs, holdings, underlyings, ratios, type, stock_price, crisis, listmap, ratio_latest, return_periods, underlying_map)


@st.cache
def price_normalise(data, date_select, type = 2):
    date_min = data.groupby('Ticker')['Date'].min().tolist()
    date_min.append(date_select)
    max_date = max(date_min)
    data_sub = data[data['Date']>= max_date]
    fac = data_sub[data_sub.groupby('Ticker')['Date'].transform(min) == data_sub['Date']]
    if type == 1:

        #data.loc[data['Date'] == max_date1,['Ticker', 'Price']]
        fac['Factor1'] = 1/fac['Price']


        norm_fac = pd.merge(data_sub, fac[['Ticker', 'Factor1']], how='left', on='Ticker')
        norm_fac['Price_norm'] = (norm_fac['Price'] * norm_fac['Factor1'])-1

        norm_display = norm_fac[['Date', 'Ticker', 'Price_norm']]
        norm_display.columns = ['Date', 'Ticker', 'Price']
    else:
        #fac = data.loc[data['Date'] == max_date1,['Ticker', 'Price', 'Total Return']]
        fac['Factor1'] = 1/fac['Price']
        fac['Factor2'] = 1/fac['Total Return']

        norm_fac = pd.merge(data_sub, fac[['Ticker', 'Factor1', 'Factor2']], how='left', on='Ticker')
        norm_fac['Price_norm'] = round((norm_fac['Price'] * norm_fac['Factor1'])-1,2)
        norm_fac['TR_norm'] = round((norm_fac['Total Return'] * norm_fac['Factor2'])-1,2)

        norm_display = norm_fac[['Date', 'Ticker', 'Price_norm', 'TR_norm']]
        norm_display.columns = ['Date', 'Ticker', 'Price', 'Total Return']
    return(norm_display)

@st.cache
def add_to_session_port(data, add_to):
    idx = data.isin(add_to)
    list_add = data[~idx]
    add_to = np.append(add_to,list_add)
    return(add_to, list_add)

@st.cache
def weight_calc(data_pd_col, weight_method):
    if weight_method == 'Market Cap':
        weightsum = data_pd_col.sum()
        weight = data_pd_col.values / weightsum
    elif weight_method == 'Equal Weight':
        weight = np.ones(len(data_pd_col))
        weight =  weight /len(data_pd_col)
    else:
        weight = 1
    return weight

@st.cache(suppress_st_warning=True, allow_output_mutation=True )
def port_calc(port_tickers, port_weight, stock_price):
    port_stock_price = stock_price[stock_price['Ticker'].isin(port_tickers)]
    port_stock_pivot = port_stock_price.pivot_table(index='Date', columns = 'Ticker', values='Price')
    port_stock_pivot.sort_values('Date', inplace=True)
    port_stock_pivot.fillna(method='ffill', inplace=True)



    port_tr_pivot = port_stock_price.pivot_table(index='Date', columns = 'Ticker', values='Total Return')
    port_tr_pivot.sort_values('Date', inplace=True)
    port_tr_pivot.fillna(method='ffill', inplace=True)

    # for tickers not exist in stock_price, re_validate tickers
    ticker_valid = port_stock_pivot.columns
    idx = port_tickers.isin(ticker_valid)

    port_stock_p = port_stock_pivot[ticker_valid]
    port_stock_num = port_stock_p.values

    port_tr_p = port_tr_pivot[ticker_valid]
    port_tr_num = port_tr_p.values


    weight_num = np.tile(port_weight[idx, np.newaxis], len(port_stock_p)).transpose()
    weight_num[port_stock_p.isna()] = 0
    port_stock_num[port_stock_p.isna()] = 0
    port_tr_num[port_tr_p.isna()] = 0
    weight_num_sum = np.sum(weight_num, axis=1)
    weight_mx = weight_num / weight_num_sum[:,np.newaxis]

    index_cal = np.ones(weight_num_sum.shape)
    tr_cal = np.ones(weight_num_sum.shape)
    shares = np.zeros(weight_mx.shape)
    shares_tr = np.zeros(weight_mx.shape)

    index_cal[0] = 100
    tr_cal[0] = 100
    for i in range(0, len(index_cal)-1):
            shares[i] = index_cal[i] * weight_mx[i] / port_stock_num[i]
            shares_tr[i] = tr_cal[i] * weight_mx[i] / port_tr_num[i]

            index_cal[i+1] = np.nansum(shares[i] * port_stock_num[i+1])
            tr_cal[i+1] = np.nansum(shares_tr[i] * port_tr_num[i+1])

    index_pd = pd.DataFrame(zip(index_cal, tr_cal), index = port_stock_p.index)
    index_pd.reset_index(inplace=True)
    index_pd['Ticker'] = 'My portfolio'
    index_pd.columns = ['Date', 'Price', 'Total Return', 'Ticker']

    return index_pd

@st.cache(suppress_st_warning=True,allow_output_mutation=True )
def perform_graph(data, price_type, display_type ):
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['Date'], empty='none')

    base = alt.Chart(data).mark_line().encode(
        x='Date:T',
        color='Ticker:N',
        )
    if display_type == 'Absolute':
        line = base.encode(y=price_type + ':Q')
    else:
        line = base.encode(y=alt.Y(price_type + ':Q', axis=alt.Axis(format='%')))

    selectors = alt.Chart(data).mark_point().encode(
        x='Date:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, price_type +':Q', alt.value(' '))
        )

    return alt.layer(
        line, selectors, points, text
    ).properties(
        height = 600
    )

@st.cache
def annualised_cal(data, price_type):
    port_display_pivot = data.pivot_table(index='Date', columns = 'Ticker', values=price_type)
    port_display_pivot.sort_values('Date', inplace=True)
    port_display_pivot.fillna(method='ffill', inplace=True)
    idx_year = port_display_pivot.index.year.values
    idx_year_end = idx_year[:-1] != idx_year[1:]
    idx_year_end = np.append(idx_year_end,True)

    port_year = port_display_pivot[idx_year_end]
    port_year_return = pd.DataFrame(port_year.values[1:,:]/port_year.values[:-1,:]-1,
                        index = port_year.index[1:], columns = port_year.columns)

    port_year_tab = port_year_return.stack().reset_index()
    port_year_tab.columns = ['Date','Ticker','Return']
    port_year_tab['Year'] = port_year_tab['Date'].dt.year

    year_current = port_year_tab['Year'].iloc[-1]
    year_beg = port_year_tab['Year'].iloc[0]

    date_current = port_year_tab['Date'].iloc[-1]
    date_beg = port_year_tab['Date'].iloc[0]

    port_year_tab['Years'] = port_year_tab['Year']
    port_year_tab.loc[port_year_tab['Year'] == year_current, 'Years'] = str(year_current) + ' YTD'

    return port_year_tab, year_current, year_beg, date_current, date_beg

@st.cache(suppress_st_warning=True)
def max_drawdown(data, price_type):
    data_p = data.pivot_table(index='Date', columns = 'Ticker', values=price_type)
    data_p.sort_values('Date', inplace=True)
    data_max = data_p.expanding(1).max()
    data_drawdown = data_p / data_max - 1
    data_drawdown.reset_index(inplace=True)
    drawdown_tab = data_drawdown.melt(id_vars=['Date'])
    drawdown_tab = drawdown_tab[~drawdown_tab['value'].isna()]
    return drawdown_tab


#---------------------------------------------------------------------------------- Initialise

etf_price, etfs, holdings, underlyings, ratios, type, stock_price, crisis, listmap, ratio_latest, return_periods, underlying_map = upload()

ratio_list = ratios['Short Name'].unique().tolist()
color_list = ['Country', 'Sector', 'Industry', 'Business Activity', 'Size']
size_list = np.append('Market Cap',return_periods.to_numpy(dtype='str').reshape(-1,1))

stock_list = underlyings[['Ticker', 'Company']]
stock_list['Field'] = stock_list['Ticker'] + ': '+stock_list['Company']
stock_list.columns = ['Ticker', 'Name', 'Field']
full_list = pd.concat([etfs, stock_list], axis=0, ignore_index=True)

q_list = ratios[['Date','Quarter']].drop_duplicates().sort_values('Date')['Quarter'].unique().tolist()

port = SessionState.get(name = 'My portfolio', tickers=[], filter_criteria = {},
                        criteria_tickers = [],
                        scatter_tickers = [],
                        performance=pd.DataFrame())

#--------------------------------------------------------- Scatter plot graph
st.markdown("<br>", unsafe_allow_html=True)
"""
# :sparkles:Thematic Playbook: [![Follow](https://img.shields.io/twitter/follow/InvestCoLab?style=for-the-badge)](https://twitter.com/InvestCoLab)
"""
with st.expander("Analyse popular thematic ETFs; explore alternative ways of building thematic portfolios.   👉 show more"):

    st.markdown("<br>", unsafe_allow_html=True)
    """
    Thematic ETFs provide investors a cost effective way of gaining exposure to technology innovations. These ETFs presents investors the great advantage of  -

    :white_check_mark: making pre-selected & well researched basket of thematic stocks readily availbale to investors """

    """:white_check_mark: diversify away the risks of allocating to any early stage tech companies by betting on a trend instead"""
    """
    However, thematic ETFs also have inherent disadvantage of -

    :negative_squared_cross_mark: the basket of stocks are often not pure play (e.g. the likes of many EV ETFs investing in traditional car companies).

    :negative_squared_cross_mark: allocating to certain undesirable stocks

    :negative_squared_cross_mark: some smaller sized (though well researched) ETFs may suffer from liquidity issues of the ETF itself, whereas underlying stocks are way more liquid.

    In all these case, investors are better off to structure their own thematic portfolio. In addition, savvy inestors might be keen to identify value or high growth stocks within the thematic spectrum.
    """
st.markdown("<br>", unsafe_allow_html=True)
"""This playbook intends to deliver 3 tasks in a ***visually*** ***simple*** way."""

""":one: **Analyse** ETFs\' holdings and underlyings stocks' fundamentals"""
""":two: **Screening** thematic stocks"""
""":three: **Build** and **test** the portfolio"""

"""Find out more about thematic ETFs research"""

"""[Thematic ETFs comparison: the pros and cons, winners and losers]() """
"""[Can we find value vs. growth play in tech ETFs?]() """
"""[How to build a thematic portfolio, a step-by-step guide]()"""

#-------------------------------------------------------------------------------- Holding Graph
st.write('-------------------------------------------')
"""
## :one:-a. ETFs' Holdings
"""
col_h1, col_h2 = st.columns(2)
val_h1 = col_h1.multiselect(
    'Selec ETFs: (slick empty space in the graph to restore to default)',
    etfs['Field'].tolist(),
    [etfs['Field'].iloc[0], etfs['Field'].iloc[2]]
)

val_h2 = col_h2.selectbox(
    'View by:',
    color_list
)

holding_select_ticker = etfs.loc[etfs['Field'].isin(val_h1), 'Ticker']
holding_select_map = pd.merge(holdings.loc[holdings['ETF Ticker'].isin(holding_select_ticker)].iloc[:,:-1],
                underlyings, how='left', on='Ticker')
holding_select_map = holding_select_map[~holding_select_map['Ticker'].isna()]
col_hh1, col_hh2 = st.columns(2)

with col_hh1:
    selector3 = alt.selection_single(encodings=['x','color'], empty='all')

    holding_bar = alt.Chart(holding_select_map).mark_bar(
        align='left'
    ).encode(
        x='ETF Ticker:O',
        y=alt.Y('sum(Weight):Q',scale=alt.Scale(domain=[0,100])),
        color=alt.condition(selector3, val_h2+':N', alt.value('lightgray')),
        tooltip = [val_h2, 'sum(Weight)'],
    ).add_selection(
        selector3
    ).properties(
        width = 600,
        height = 400
    )
with col_hh2:
    slider_page = alt.binding_range(min=1, max=len(holding_select_map)/20, step=1, name='Number of holdings (20/page):')
    selector_page = alt.selection_single(name="PageSelector", fields=['page'],
                                    bind=slider_page, init={'page': 1})

    ticker_bar2 = alt.Chart(holding_select_map).transform_filter(
        selector3
    ).mark_bar(align='left').encode(
        y = alt.Y('Company',sort='-x', title=None),
        x = 'sum(Weight):Q',
        color = 'ETF Ticker:N',
        tooltip = ['Ticker','Company', 'ETF Ticker','Weight'],
    ).transform_window(
        rank = 'rank(sum(Weight))',
    ).add_selection(
        selector_page
    ).transform_filter(
        '(datum.rank > (PageSelector.page - 1) * 20) & (datum.rank <= PageSelector.page * 20)'
    ).properties(
        width = 'container'
    )

st.altair_chart(holding_bar | ticker_bar2, use_container_width=True)


#--------------------------------------------------------------------------------Header
"""
---
## :one:-b. Risk, return and fundamentals
"""
val_s1 = st.multiselect(
    'Select ETFs to view all the holdings, or select stocks directly',
    full_list['Field'].tolist(),
    [full_list['Field'].iloc[0],full_list['Field'].iloc[10]]
    )

col_p1, col_p2, col_p3, col_p4 = st.columns([1,1,1,3])

val_p3 = col_p1.select_slider(
    'As of:',
    q_list,
    q_list[-1]
    )

val_p1 = col_p2.selectbox(
    'Risk/Return timeframe:',
    return_periods.index.tolist()
)
val_p2 = col_p3.radio(
    'Return type:',
    ['Cumulative', 'Annualised']
)

val_p4 = col_p4.radio(
    'Price type',
    ['Price', 'Total Return: with dividend reinvested']
)

if val_p4 != 'Price':
    val_p4 = 'Total Return'

return_select = return_periods.loc[val_p1, val_p2]
vol_select = return_periods.loc[val_p1,'Volatility']

etf_ticker = etfs.loc[etfs['Field'].isin(val_s1),'Ticker']
stock_ticker = stock_list.loc[stock_list['Field'].isin(val_s1),'Ticker']
holding_select = holdings.loc[holdings['ETF Ticker'].isin(etf_ticker)]

holding_ticker = holding_select['Ticker'].drop_duplicates()
stock_ticker_all = pd.concat([stock_ticker, holding_ticker], axis=0, ignore_index=True)


ratio_select1 = ratios[(ratios['Ticker'].isin(stock_ticker_all)) & (ratios['Quarter'] == val_p3) &
                    ((ratios['Short Name'] == return_select) | (ratios['Short Name'] == vol_select))]
ratio_p1 = ratio_select1.pivot_table(index = 'Ticker', columns = 'Short Name', values='Value').reset_index()

# Concat ETF price and Stock price
price_select = pd.concat([etf_price[etf_price['Ticker'].isin(etf_ticker)],
                stock_price[stock_price['Ticker'].isin(stock_ticker_all)]], axis=0, ignore_index=True)
price_select_norm = price_normalise(price_select, pd.to_datetime('1995-01-01'), 2)

#--------------------------------------------------------------------------------- Fundamental 1 Graph
selector1 = alt.selection_single(name='dot_select', empty='all', fields=['Ticker'])
base = alt.Chart(ratio_p1).properties(
            width = 600
            ).mark_point(filled=True, size=200).encode(
                x=alt.X(vol_select+':Q',scale=alt.Scale(zero=False)),
                y=alt.Y(return_select+':Q',scale=alt.Scale(zero=False), axis=alt.Axis(format='%')),
                color=alt.condition(selector1, 'Ticker:N', alt.value('lightgray'), legend=None),
            )
points = base.encode(
            tooltip = [alt.Tooltip('Ticker:N'), alt.Tooltip(return_select, format='%'),alt.Tooltip(vol_select, format='.2f')],
        ).add_selection(
            selector1
        ).properties(
            title='Risk/return efficient frontier'
        ).mark_text(align='left', baseline='top').encode(
        text='Ticker')

polynomial_fit = base.transform_regression(
        vol_select, return_select, method="poly", order=2, as_=[vol_select, str(2)]
    ).mark_line().encode(color = alt.value('lightgray')).transform_fold([str(2)], as_=["degree", return_select])

timeseries = alt.Chart(price_select_norm).mark_line().encode(
    x='Date',
    y=alt.Y(val_p4+':Q',axis=alt.Axis(format='%')),
    color=alt.condition('datum.Ticker == dot_select.Ticker', 'Ticker:N', alt.value('lightgray'),legend=None),
    tooltip = [alt.Tooltip('Ticker'), alt.Tooltip('Price', format='%')]
).add_selection(
    selector1
).properties(
    title = 'Historial performance',
    width=600
)

points + alt.layer(base, polynomial_fit)| timeseries
#--------------------------------------------------------------------------------- Fundamental 2 Calc

col_r0, col_r1, col_r2, col_r3, col_r4 = st.columns(5)

val_r0 = col_r0.select_slider(
    'As of:',
    q_list[:-1],
    q_list[-2],
    key=2
    )

val_r1 = col_r1.selectbox(
    'Y axis ratio:',
    ratio_list,
    24
)
idx_r1 = ratios['Short Name'] == val_r1

val_r2 = col_r2.selectbox(
    'X axis ratio:',
    ratio_list,
    41
)
idx_r2 = ratios['Short Name'] == val_r2

val_r3 = col_r3.selectbox(
    'Color by:',
    color_list
)
idx_r3 = ratios['Short Name'] == val_r3

val_r4 = col_r4.selectbox(
    'Size by:',
    size_list
)
idx_r4 = ratios['Short Name'] == val_r4

#--------------------------------------------------------------------------------- Fundamental 2 Graph

ratio_select2 = ratios[(ratios['Ticker'].isin(stock_ticker_all)) &
                    (idx_r1| idx_r2 | idx_r3 | idx_r4)]
ratio_select2_snap = ratio_select2[ratio_select2['Quarter'] == val_r0]
ratio_p2 = ratio_select2_snap.pivot_table(index = 'Ticker', columns = 'Short Name', values='Value').reset_index()
ratio_p2_join = pd.merge(ratio_p2, underlyings, how='left', on='Ticker')

selector2 = alt.selection_single(name='r_select', empty='all', fields=['Ticker'])
base2 = alt.Chart(ratio_p2_join).add_selection(
    selector2
)
ratio_points = base2.mark_point(filled=True, size=200).encode(
    x=val_r1+':Q',
    y=val_r2+':Q',
    color=alt.condition(selector2, val_r3+':N', alt.value('lightgray'), legend=None),
    size = alt.Size(val_r4 + ':N',legend=None),
    tooltip = [alt.Tooltip('Ticker:N'), alt.Tooltip('Company:O'),
                alt.Tooltip(val_r1, format='.2f'), alt.Tooltip(val_r2, format='.2f')]
).properties(
    width = 600,
    title = 'Fundamentals plot'
)

#ratio_select2.sort_values(['Ticker', 'Date'], ascending=True, inplace=True)
ratio_select2_t = ratio_select2.pivot_table(index = ['Ticker','Quarter'], columns = 'Short Name', values='Value').reset_index()
base = alt.Chart(ratio_select2_t).mark_bar().encode(
    x=alt.X('Quarter:O', sort=q_list, title=None),
    color = alt.condition('datum.Ticker == r_select.Ticker','Ticker:O',alt.value('lightgrey'), legend=None),
    tooltip = ['Ticker']
).add_selection(
    selector2
).transform_filter(
    selector2
).properties(
    width=600,
    height = 110
)

ratio_points | (base.encode(y=alt.Y(val_r1, stack=None, title=None)).properties(title = 'Historical ' + val_r1) & base.encode(y=alt.Y(val_r2,stack=None, title=None)).properties(title='Historial '+val_r2))


#------------------------------------------------------------------------------- Portfolio construction
st.write('----')
col_c1, col_c2 = st.columns([4,1])
with col_c1:
    """## :two: Screening thematic stocks"""
with col_c2:
    val_c2 = col_c2.number_input(
        'Add number of filters',
        1, 10, 3, 1
    )

val_c1 = st.multiselect(
    'Universe',
    etfs['Field'].tolist(),
    [etfs['Field'].iloc[0], etfs['Field'].iloc[2]]
)

@st.cache
def underlying_select_calc(multi_select):
    ticker_univere = holdings.loc[holdings['ETF Ticker'].isin(etfs.loc[etfs['Field'].isin(multi_select),'Ticker']),'Ticker'].unique()
    underlying_s = underlying_map[underlying_map['Ticker'].isin(ticker_univere)]
    return underlying_s

@st.cache(allow_output_mutation=True)
def filter_bar_graph(data, filter_val):
    return alt.Chart(data).mark_bar().encode(
        x=alt.X(filter_val+':N', sort='-y',axis=None),
        y=alt.Y('count()', title='count'),
        color = alt.condition('datum.Color_bool', 'Color_bool', alt.value('lightgray'), legend=None),
        tooltip = [filter_val, 'count()']
        ).properties(
            height = 100
        )

@st.cache(allow_output_mutation=True)
def filter_hist_graph(data, filter_val):
    return alt.Chart(data).mark_bar().encode(
        x = alt.X(filter_val+ ':Q', bin=alt.Bin(maxbins=20)),
        y = alt.Y('count():Q', title='Distribution'),
        color = alt.condition('datum.Color_bool', 'Color_bool', alt.value('lightgray'), legend=None),
        tooltip = ['count()']
    ).properties(
       height = 100
    )

underlying_select = underlying_select_calc(val_c1)

filters = {}
idx = {}
filter_graph = {}
filter_data = {}

for i in range(1, val_c2+1):
    col1, col2, col3 = st.columns([1,2,2])
    filter_item = col1.selectbox(
            'Filter:',
            np.append(color_list, ratio_list),
            i*2-2,
            key = i
        )
    filter_data[i] = underlying_select[['Ticker',filter_item]]

    if filter_item in color_list:
        ls = listmap[filter_item].tolist()
        filter_val_i = col2.multiselect(
                'Select:',
                ls,
                [ls[0],ls[1]],
                key = i
            )

        if filter_val_i:
            filters[filter_item] = filter_val_i
            idx[i] = underlying_select[filter_item].isin(filter_val_i)
            filter_data[i]['Color_bool'] = idx[i]

        with col3:

            bar_graph = filter_bar_graph(filter_data[i], filter_item)
            filter_graph[i] = bar_graph
            st.altair_chart(filter_graph[i], use_container_width=True)

    else:
        ls = listmap['Ratio'].loc[filter_item].values.tolist()

        filter_val_i = col2.slider(
                '',
                ls[0], ls[1], (ls[0],ls[1]), ls[2],
                key = i
            )

        if filter_val_i:
                filters[filter_item] = filter_val_i
                idx[i] = (underlying_select[filter_item]>= filter_val_i[0] ) & (underlying_select[filter_item] <= filter_val_i[1] )
                filter_data[i]['Color_bool'] = idx[i]
        with col3:
             hist_graph = filter_hist_graph(filter_data[i], filter_item )
             filter_graph[i] = hist_graph
             col3.altair_chart(filter_graph[i], use_container_width=True)

    idx_full = np.array(np.ones(underlying_select.shape[0]), dtype=bool)
    for i in idx.keys():
        idx_full = idx_full & idx[i]
    underlying_filter = underlying_select.loc[idx_full]
    underlying_filter.index = [x for x in range(1, len(underlying_filter)+1)]
    port.criteria_tickers = underlying_filter['Ticker'].values

#------------------------------------------------------------------------- Display Filter
col_names = ['Ticker', 'Company', 'Country', 'Sector']

@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def scatter_brush(data, y, x):
    brush = alt.selection(type='interval', name = 'filter_select',resolve='global')
    text_base = alt.Chart(data).mark_text(align='left', baseline='top').encode(
                    y= y + ':Q',
                    x = alt.X(x + ':Q',scale=alt.Scale(clamp=True)),
                    color=alt.condition(brush, 'Ticker:N', alt.ColorValue('gray'), legend=None),
                    tooltip=[ 'Ticker', alt.Tooltip(y, format='.2f'), alt.Tooltip(x, format='.2f')],
                    text='Ticker'
            ).add_selection(
                brush
            ).properties(
                width = 600,
                height = 250,
                title = 'Filtered results (drag & select to choose stocks)'
            )

    return text_base

with st.expander('Filter results ('+ str(len(underlying_filter)) + ')        👉  click ➕/➖ to show or hide the filter results', expanded=True):
    col_x1, col_x2 = st.columns(2)
    col_x1, col_x3, col_x4 = st.columns([2,1,1])
    col_x1.subheader('')
    add_placeholder = col_x1.empty()
    val_x3 = col_x3.selectbox(
        'Y axis',
        ratio_list,
        0
    )
    val_x4 = col_x4.selectbox(
        'X axis',
        ratio_list,
        1
    )

    col_l1, col_l2 = st.columns(2)

    with col_l2:
        s_select = altair_component(scatter_brush(underlying_filter[['Ticker', 'Company', val_x3, val_x4]], val_x3, val_x4))
        if len(s_select) > 1:
            lower1 = round(s_select[val_x3][0],2)
            upper1 = round(s_select[val_x3][1],2)
            lower2 = round(s_select[val_x4][0],2)
            upper2 = round(s_select[val_x4][1],2)
            select_display = underlying_filter.loc[(underlying_filter[val_x3] >= lower1) & (underlying_filter[val_x3] <= upper1) &
                                    (underlying_filter[val_x4] >= lower2) & (underlying_filter[val_x4] <= upper2)]
            col_l1.write(str(len(select_display)) + ' is selected')
        else:
            select_display = underlying_filter

    select_display.index = [x for x in range(1, len(select_display)+1)]
    col_l1.write(select_display[col_names])

    if add_placeholder.button('⚡ Add all to portfolio') :
        port.tickers, ticker_added = add_to_session_port(select_display['Ticker'],port.tickers)
        if len(ticker_added) == len(select_display):
            st.info(str(len(ticker_added)) + ' is added to the portfolio')
        else:
            st.info(str(len(ticker_added)) + ' is added to the portfolio, ' + str(len(select_display) - len(ticker_added)) + ' stocks already exist in the portfolio')

st.markdown("<br>", unsafe_allow_html=True)
#-------------------------------------------------------------------------------- Portfolio
st.markdown("<br>", unsafe_allow_html=True)
"""## :three: Build and test the portfolio"""
col_f1, col_f2, col_f3 = st.columns([2,3,1])

val_f1 = col_f1.text_input('Portfolio name:', port.name)
port.name = val_f1

val_f2 = col_f2.multiselect(
        'Add stocks or ETFs to portfolio',
        stock_list['Field'].tolist()
    )

col_f3.title('')
if col_f3.button('Add to portfolio'):
    port.tickers, ticker_added = add_to_session_port(stock_list.loc[stock_list['Field'].isin(val_f2),'Ticker'],port.tickers)
    st.info(str(len(ticker_added)) + 'has been added')

port_empty = False
col_d1, col_d2, col_d3 = st.columns(3)
placeholder_d1 = col_d1.empty()
placeholder_d2 = col_d2.empty()
placeholder_d3 = col_d3.empty()
port_placeholder = st.empty()
button_placeholder = st.empty()

if len(port.tickers)>0:
    val_d1 = placeholder_d1.selectbox(
                    'Weight:',
                    ['Market Cap', 'Equal Weight', 'Fundamental Weight']
                )

    val_d2 = placeholder_d2.selectbox(
                    'Dividend',
                    ['Reinvest', 'Accumulate']
                )

    val_d3 = placeholder_d3.selectbox(
                    'Rebalance',
                    ['Quarterly', 'Semi-Annual', 'Annual']
                )

    col_port = ['Ticker', 'Company', 'Weight', 'Country', 'Sector', 'IPO Date']
    port_struc = underlying_map[underlying_map['Ticker'].isin(port.tickers)]
    port_struc['Weight'] = weight_calc(port_struc['Market Cap'], val_d1)
    port_struc.index = [x for x in range(1, len(port_struc)+1)]

        # Print portfolio
    port_placeholder.table(port_struc[col_port])

    index_output = port_calc(port_struc['Ticker'], port_struc['Weight'], stock_price)
    index_port = index_output.copy()
    index_port['Ticker'] = port.name
    port.performance = index_port

    port_empty = button_placeholder.button('🗑️ Empty portfolio')
    st.write('------------------------------------------')

if port_empty:
    port.tickers = []
    placeholder_d1.empty()
    placeholder_d2.empty()
    placeholder_d3.empty()
    port_placeholder.info('The portfolio is emptied')
    button_placeholder.empty()

        #------------------------------------------------------------------------- Portfolio performance
if len(port.tickers) > 0:
    with st.container():
        col_s1, col_s2, col_s3 , col_s4, col_s5 = st.columns([2,1,1,1,2])
        val_s1 = col_s1.radio(
                'Periods:',
                np.append('Customise period', crisis.index)
            )

        val_s2_from = col_s2.date_input (
                    'From date:',
                    index_port['Date'].iloc[0],
                    min_value = index_port['Date'].iloc[0],
                    max_value = index_port['Date'].iloc[-1]
                )

        val_s2_to = col_s3.date_input (
                    'From date:',
                    pd.to_datetime('2021-01-07'),
                    #datetime.date(2021,1,7),
                    min_value = index_port['Date'].iloc[0],
                    max_value = index_port['Date'].iloc[-1]
                )

        val_s3 = col_s4.radio(
                'Price type',
                ('Absolute', 'Normalised')
            )

        val_s4 = col_s5.multiselect(
                'Add to compare',
                full_list['Field'].tolist(),
                [full_list['Field'].iloc[0],full_list['Field'].iloc[10]]
            )

        if val_s1 == 'Customise period':
            lower_date = pd.to_datetime(val_s2_from)
            upper_date = pd.to_datetime(val_s2_to)

        else:
            lower_date = crisis.loc[val_s1,'Start Date']
            upper_date = crisis.loc[val_s1,'End Date']

        perf_etf_ticker = etfs.loc[etfs['Field'].isin(val_s4),'Ticker']
        perf_stock_ticker = stock_list.loc[stock_list['Field'].isin(val_s4),'Ticker']

        perf = pd.concat([index_port, etf_price[etf_price['Ticker'].isin(perf_etf_ticker)],
                            stock_price[stock_price['Ticker'].isin(perf_stock_ticker)]], axis=0, ignore_index=True)

        perf_norm = price_normalise(perf[(perf['Date'] >= lower_date) &(perf['Date'] <= upper_date)], lower_date, type=2)


        perf_graph = perform_graph(perf_norm, 'Price', 'Normalised')

        st.altair_chart(perf_graph, use_container_width=True)

            #---------------------------------------------------------------------- annualised
        st.write('----------')
        port_annualised, current, beg, current_day, beg_day = annualised_cal(perf, 'Price')
        col_a0, col_a1, col_a2 = st.columns(3)
        year_map = pd.DataFrame({'Period Num': [current-3, current-5, current-10, beg ],
                                        'Year count': [3, 5, 10, current - beg],
                                        'Date Num': [current_day - DateOffset(years=3), current_day - DateOffset(years=5), current_day - DateOffset(years=10), beg_day]},
                                index = ['3Y', '5Y', '10Y', 'All'])

        col_a0.write('')
        col_a0.subheader('Calendar performnace')
        year_slider = col_a1.select_slider(
                        'Number of years',
                        year_map.index.tolist(),
                        '5Y'
                        )

        val_c2 = col_a2.radio(
                'Return type',
                ['Calendar Year', 'Cumulative', 'Annualised']
            )
        year_select = year_map.loc[year_slider, 'Period Num']
        port_year_display = port_annualised[port_annualised['Date'].dt.year>=year_select]
        port_year_display.sort_values('Date', ascending=False, inplace=True)

        port_year_bar = alt.Chart(port_year_display).mark_bar().encode(
                                x='Ticker:N',
                                y=alt.Y('Return:Q',axis=alt.Axis(format='%')),
                                color='Ticker:N',
                                column='Years:O',
                                tooltip = ['Ticker', 'Return:Q']
                                ).properties(
                                    width = 1000/year_map.loc[year_slider, 'Year count']
                                )

        st.altair_chart(port_year_bar, use_container_width=True)


        st.write('----------')
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.write('')
        col_w1.subheader('Max drawdowns')
        year_slider_drawdown = col_w2.select_slider(
                        'Number of years',
                        year_map.index.tolist(),
                        '5Y',
                        key = 2
                        )

        val_c2 = col_w3.radio(
                'Return type',
                ['Total Return', 'Price Return']
            )



        date_select = year_map.loc[year_slider_drawdown, 'Date Num']
        drawdown = max_drawdown(perf[perf['Date']>=date_select], 'Price')

        drawdown_graph = perform_graph(drawdown, 'value','Normalised')
        st.altair_chart(drawdown_graph, use_container_width=True)

else:
    st.warning('Portfolio is empty. Add stocks one-by-one or add stocks in bulk from the filter list')
