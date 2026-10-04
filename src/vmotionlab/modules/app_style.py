import streamlit as st


def apply_global_style() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: dark; }
        .vmotion-title {
            font-size: 2.35rem;
            font-weight: 750;
            margin-top: .2rem;
            color: #F8FAFC;
        }
        .vmotion-subtitle {
            font-size: 1.12rem;
            color: #A7B0BE;
            margin-bottom: 1rem;
        }
        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
        }
        .workflow-card {
            border-radius: 0.7rem;
            padding: 0.72rem 0.35rem;
            text-align: center;
            min-height: 4.35rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
            gap: 0.18rem;
        }
        .workflow-ready {
            color: #BFDBFE;
            background: rgba(37, 99, 235, 0.20);
            border: 1px solid #3B82F6;
        }
        .workflow-pending {
            color: #FECACA;
            background: rgba(220, 38, 38, 0.18);
            border: 1px solid #EF4444;
        }
        .workflow-name {
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.1;
        }
        .workflow-state {
            font-size: 0.78rem;
            font-weight: 600;
            opacity: 0.95;
        }
        div[data-testid="stMetric"] {
            background: #161B22;
            border: 1px solid #30363D;
            border-radius: 0.65rem;
            padding: 0.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
