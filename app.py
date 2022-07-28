from re import A
import streamlit as st
import numpy as np
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import stats
import tkinter as tk

plt.style.use('grayscale')
plt.style.use('seaborn-whitegrid')
np.random.seed(0)

# ページの既定の設定を構成する
st.set_page_config(page_title="ベイズ統計×A/Bテスト",
                   page_icon="👩‍💻",
                   initial_sidebar_state="collapsed",
                   )

st.set_option('deprecation.showPyplotGlobalUse', False)

# 関数の実行をメモするための関数デコレータ
# @st.cache(persist=False,
#           allow_output_mutation=True,
#           suppress_st_warning=True,
#           show_spinner= True)

# 0以上1以下の範囲を1001分割した配列thetasを用意する
thetas = np.linspace(0, 1, 1001)

#尤度関数likelihoodを用意
likelihood = lambda r: thetas if r else (1 - thetas)

#事後分布を算出する関数
def posterior(a, N):
    alpha = a + 1
    beta = N - a + 1
    numerator = thetas ** (alpha - 1) * (1 - thetas) ** (beta - 1)
    return numerator / numerator.sum()

# 事前分布を一様分布にする
p = np.array([1 / len(thetas) for _ in thetas])

st.title("A/Bテスト")
st.write("このアプリは、ベイズ統計を用いてA/Bテストを行います。")

st.subheader('1.データの入力')
st.write('A案とB案のクリック数・インプレッション数を入力してください。')

with st.container():
    with st.expander("A案クリック数"):
        A_clicks = st.number_input("クリックされた数を入力してください。", min_value=0,key='A_clicks')
        
    with st.expander("A案インプレッション数"):
        A_impression = st.number_input("インプレッション数の値を入力してください。", min_value=0,key='A_impression')

with st.container():
    with st.expander("B案クリック数"):
        B_clicks = st.number_input("クリックされた数を入力してください。", min_value=0,key='B_clicks')
        
    with st.expander("B案インプレッション数"):
        B_impression = st.number_input("インプレッション数の値を入力してください。", min_value=0,key='B_impression')

def hmv(xs, ps, alpha=0.95):
    xps = sorted(zip(xs, ps), key=lambda xp: xp[1], reverse=True)
    xps = np.array(xps)
    xs = xps[:, 0]
    ps = xps[:, 1]
    return np.sort(xs[np.cumsum(ps) <= alpha])

def plot_hdi(ps, label):
    hm_thetas = hmv(thetas, ps, 0.95)
    plt.plot(thetas, ps)
    plt.annotate('', xy=(hm_thetas.min(), 0),
                    xytext=(hm_thetas.max(), 0),
                    arrowprops=dict(color='black', shrinkA=0, shrinkB=0,
                                    arrowstyle='<->', linewidth=2))
    plt.annotate('%.3f' % hm_thetas.min(), xy=(hm_thetas.min(), 0),
                    ha='right', va='bottom')
    plt.annotate('%.3f' % hm_thetas.max(), xy=(hm_thetas.max(), 0),
                    ha='left', va='bottom')
    hm_region = (hm_thetas.min() < thetas) & (thetas < hm_thetas.max())
    plt.fill_between(thetas[hm_region], ps[hm_region], 0, alpha=0.3)
    plt.xlim(0, 0.3)
    plt.ylabel(label)
    plt.yticks([])


st.subheader('2.事後分布の算出')
st.write('入力されたデータをもとに、A案B案の事後分布の算出を行います。')
    
if A_clicks and A_impression and B_clicks and B_impression:
    if st.checkbox("算出",key="calculation"):
        st.write('グレーの部分は95%HDI(highest density interva)の区間となります。区間の両端にある数値はその区間の最小値と最大値を表します。')
        st.write('解釈の例')
        st.write('仮説を「B案のクリック率は、A案よりも大きい。」とした場合、B案のクリック率の95%HDI全体がA案のクリック率の95%HDIの外にあるとき、2つのクリック率は異なるといえる。特に、B案のHDIの最小値がA案のHDIの最大値より大きいとき、B案のクリック率はA案よりも大きいといえる（反対の場合も同様）。それ以外の場合は、結論づけられない。')
        fig1 = plt.subplot(2, 1, 1)
        A = posterior(A_clicks, A_impression)
        plot_hdi(A, 'A')
        fig2 = plt.subplot(2, 1, 2)
        B = posterior(B_clicks, B_impression)
        plot_hdi(B, 'B')
        fig1 = plt.xlabel(r'$\theta$')
        fig1 = plt.tight_layout()
        fig2 = plt.xlabel(r'$\theta$')
        fig2 = plt.tight_layout()
        st.pyplot(fig1)
        st.pyplot(fig2)
        



        


