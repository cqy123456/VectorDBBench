def hideSidebar(st):
    st.markdown(
        """<style>
            div[data-testid='collapsedControl'] {display: none;}
        </style>""",
        unsafe_allow_html=True,
    )


def wideContent(st):
    st.markdown(
        """<style>
            .block-container { max-width: 1080px; }
        </style>""",
        unsafe_allow_html=True,
    )
